import Foundation
import llama
import OSLog

public typealias Token = llama_token
public typealias Model = OpaquePointer
public typealias Chat = (role: Role, content: String)

@globalActor public actor InferenceActor {
    public static let shared = InferenceActor()
}

open class LLM: ObservableObject {
    private let logger = Logger(subsystem: "swift.LLM", category: "LLM")

    public var model: Model
    public var preProcess: (_ history: [Chat]) -> String = { $0.map(\.content).joined() }
    public var postProcess: (_ output: String) -> Void = { _ in }
    public var update: (_ outputDelta: String?) -> Void = { _ in }
    public var template: Template {
        didSet {
            preProcess = template.preProcess
            stopSequences = template.stopSequences
        }
    }

    public var topK: Int32
    public var topP: Float
    public var temp: Float
    public var repeat_last_n: Int
    public var repeat_penalty: Float
    public var presence_penalty: Float
    public var frequency_penalty: Float
    public var path: URL

    private var context: Context?
    private var batch: llama_batch
    private let maxTokenCount: Int
    private let totalTokenCount: Int
    private let newlineToken: Token
    private var stopSequences: [String] = []
    private var params: llama_context_params
    private var isFull = false
    private var updateProgress: (Double) -> Void = { _ in }
    private var shouldContinuePredicting = false
    private var currentCount: Int32 = 0

    public init(
        from path: URL,
        template: Template,
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        repeat_penalty: Float = 1.1,
        repeat_last_n: Int = 64,
        presence_penalty: Float = 0.0,
        frequency_penalty: Float = 0.0,
        maxTokenCount: Int32 = 2048
    ) {
        self.path = path
        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
            modelParams.n_gpu_layers = 0
        #endif
        let cPath = path.path.cString(using: .utf8)
        guard let model = llama_load_model_from_file(cPath, modelParams)
        else { fatalError("Failed to load model from file") }
        params = llama_context_default_params()
        let processorCount = UInt32(ProcessInfo().processorCount)
        self.maxTokenCount = Int(min(maxTokenCount, llama_n_ctx_train(model)))
        params.seed = seed
        params.n_ctx = UInt32(maxTokenCount)
        params.n_batch = params.n_ctx
        params.n_threads = processorCount
        params.n_threads_batch = processorCount
        self.topK = topK
        self.topP = topP
        self.temp = temp
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.model = model
        self.totalTokenCount = Int(llama_n_vocab(model))
        self.newlineToken = model.newLineToken
        self.template = template
        self.stopSequences = template.stopSequences
        self.preProcess = template.preProcess
        batch = llama_batch_init(Int32(self.maxTokenCount), 0, 1)
        logger.debug("PARAMS: \(self.params)")
    }

    deinit {
        llama_free_model(model)
    }

    public convenience init(
        from huggingFaceModel: HuggingFaceModel,
        to url: URL = .documentsDirectory,
        as name: String? = nil,
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        maxTokenCount: Int32 = 2048,
        updateProgress: @escaping (Double) -> Void = { print(String(format: "downloaded(%.2f%%)", $0 * 100)) }
    ) async throws {
        let url = try await huggingFaceModel.download(to: url, as: name) { progress in
            Task { await MainActor.run { updateProgress(progress) } }
        }
        self.init(
            from: url,
            template: huggingFaceModel.template,
            seed: seed,
            topK: topK,
            topP: topP,
            temp: temp,
            maxTokenCount: maxTokenCount
        )
        self.updateProgress = updateProgress
    }

    /// Sets flag to stop predicting.
    public func stop() {
        shouldContinuePredicting = false
    }

    /// Resets the seed to a new random value.
    public func setNewSeed() {
        params.seed = UInt32.random(in: .min ... .max)
    }

    @InferenceActor
    private func predictNextToken() async -> Token {
        guard let context else { fatalError("Context is nil") }
        let logits = llama_get_logits_ith(context.pointer, batch.n_tokens - 1)!
        var candidates = (0 ..< totalTokenCount).map { token in
            llama_token_data(id: Int32(token), logit: logits[token], p: 0.0)
        }
        var token: llama_token!
        candidates.withUnsafeMutableBufferPointer { pointer in
            var candidates = llama_token_data_array(
                data: pointer.baseAddress,
                size: totalTokenCount,
                sorted: false
            )
            llama_sample_top_k(context.pointer, &candidates, topK, 1)
            llama_sample_top_p(context.pointer, &candidates, topP, 1)
            llama_sample_temp(context.pointer, &candidates, temp)
            // llama_sample_repetition_penalties(ctx: OpaquePointer!, candidates: UnsafeMutablePointer<llama_token_data_array>!, last_tokens: UnsafePointer<llama_token>!, penalty_last_n: Int, penalty_repeat: Float, penalty_freq: Float, penalty_present: Float)
            llama_sample_repetition_penalties(context.pointer, &candidates, batch.token, repeat_last_n, repeat_penalty, frequency_penalty, presence_penalty)
            token = llama_sample_token(context.pointer, &candidates)
        }
        batch.clear()
        batch.add(token, currentCount, [0], true)
        context.decode(batch)
        return token
    }

    /// - Returns: `true` if ready to predict the next token, `false` otherwise.
    private func prepare(history: [Chat]) -> Bool {
        var start = DispatchTime.now()
        guard !history.isEmpty else { return false }
        context = .init(model, params)
        guard let context else { fatalError("Context is nil") }

        let tokens = truncateAndEncode(history: history)
        logger.debug("Truncated and encoded in \(String(format: "%.2f", Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000)) seconds")
        start = DispatchTime.now()

        let initialCount = tokens.count
        currentCount = Int32(initialCount)
        for (i, token) in tokens.enumerated() {
            batch.n_tokens = Int32(i)
            batch.add(token, batch.n_tokens, [0], i == initialCount - 1)
        }
        logger.debug("Added tokens to batch in \(String(format: "%.2f", Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000)) seconds")
        start = DispatchTime.now()
        context.decode(batch)
        logger.debug("Decoded batch in \(String(format: "%.2f", Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000)) seconds")
        shouldContinuePredicting = true
        return true
    }

    /// Returns list of tokens, encoded from the history, truncated to the maximum token count.
    private func truncateAndEncode(history: [Chat]) -> [Token] {
        var tokens = encode(preProcess(history))
        guard tokens.count > maxTokenCount else { return tokens }

        // Truncate content from the start of history and recount tokens
        var truncatedHistory = history

        /// Buffer to add (remove, really) to the max token count to enforce more truncation
        var buffer = Double(maxTokenCount) * 0.10

        /// Number of times we've tried to truncate the history
        var tries = 0

        // While the tokens count is greater than the max token count, and we haven't tried to truncate the history more than 5 times
        var index = 0
        while tokens.count > maxTokenCount, index < truncatedHistory.count, tries <= 5 {
            // Get the amount of content we need to remove
            let tokensToRemove = (tokens.count - maxTokenCount) + Int(buffer)
            let contentToRemove = min(
                tokensToRemove * 4,
                truncatedHistory[index].content.count
            ) // Multiply by 4 for average token

            if contentToRemove == truncatedHistory[index].content.count {
                truncatedHistory.remove(at: index)
                logger.debug("Removed all of chat \(index)")
                index -= 1
            } else {
                // Remove the content
                truncatedHistory[index].content.removeFirst(contentToRemove)
                logger.debug("Removed \(contentToRemove) characters from chat \(index)")
            }

            // Update tokens
            tokens = encode(preProcess(truncatedHistory))
            logger.debug("Tokens: \(tokens.count)")
            index += 1

            // If we've reached the end of the history, increase buffer and start again.
            if index >= truncatedHistory.count {
                tries += 1
                index = 0
                buffer += Double(maxTokenCount) * 0.10
            }
        }

        if tokens.count > maxTokenCount {
            logger.error("WARNING: Truncation failed, tokens count is still greater than max token count. \(tokens.count) > \(self.maxTokenCount)")
        }

        return tokens
    }

    /// - Returns: `true` if the we should continue predicting based on the token, `false` otherwise, if it's the stop token or if we're not supposed to continue predicting.
    private func process(_ token: Token, to output: borrowing AsyncStream<String>.Continuation) -> Bool {
        // Static variables to preserve state across multiple invocations of this function
        enum saved {
            static var stopSequenceEndIndices: [Int] = [] // Tracks the current index in the stop sequence
            static var letters: [CChar] = [] // Temporarily stores letters leading up to the stop sequence
        }

        // Early return if the token matches the model's end token
        guard token != model.endToken else { return false }

        // Early return if we're not supposed to continue predicting
        guard shouldContinuePredicting else { return false }

        // Decodes the token into a string (word)
        var word = decode(token)

        // If there's a stop sequence defined, proceed with additional checks
        guard stopSequences.count != 0 else { output.yield(word); return true }

        // Convert stop sequences to CStrings and store their lengths
        let stopSequences = stopSequences.map(\.utf8CString)
        let stopSequenceLengths = stopSequences.map { $0.count - 1 }
        // Initialize stop sequence end indices if not already initialized, just a bunch of 0s
        saved.stopSequenceEndIndices = Array(repeating: 0, count: stopSequences.count)

        // Iterate through the word and check if it matches any of the stop sequences
        var found = false
        var letters: [CChar] = []

        for letter in word.utf8CString {
            guard letter != 0 else { break }

            for (index, stopSequence) in stopSequences.enumerated() {
                // If the current letter matches the stop sequence at the current index, increment the index
                if letter == stopSequence[saved.stopSequenceEndIndices[index]] {
                    saved.stopSequenceEndIndices[index] += 1
                    found = true
                    saved.letters.append(letter)

                    if saved.stopSequenceEndIndices[index] == stopSequenceLengths[index] {
                        // Stop sequence matched, perform reset and return
                        saved.stopSequenceEndIndices[index] = 0
                        saved.letters.removeAll()
                        logger.debug("Stop sequence found: \(word)")
                        return false
                    }
                } else {
                    saved.stopSequenceEndIndices[index] = 0
                }
            }

            // If the letter didn't match any stop sequence, append it to the letters array
            if !found {
                letters.append(letter)
            }
        }

        if found {
            // Handle case where letters matched part of a stop sequence
            // If the sequence was being matched but the current letter doesn't fit, reset and output
            saved.stopSequenceEndIndices = Array(repeating: 0, count: stopSequences.count)
            if !saved.letters.isEmpty {
                word = String(cString: saved.letters + [0]) + word // Prepend any matched letters to the word
                saved.letters.removeAll()
            }
            logger.debug("Stop sequence found: \(word)")
            output.yield(word) // Output the word to the stream
        } else {
            // No stop sequence found, output the entire word
            output.yield(word)
        }

        return true
    }

    private func getResponse(from history: borrowing[Chat]) -> AsyncStream<String> {
        .init { output in Task {
            var start = DispatchTime.now()
            guard prepare(history: history) else { return output.finish() }
            logger.debug("Prepared in \(String(format: "%.2f", Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000)) seconds")

            var outputString = ""

            while currentCount < maxTokenCount {
                start = DispatchTime.now()
                let token = await predictNextToken()
                // logger.debug("Predicted token in \(String(format: "%.2f", Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000)) seconds")
                if !process(token, to: output) { return output.finish() }
                currentCount += 1

                start = DispatchTime.now()
                outputString += decode(token)

                // Now we check if we have to slide the history window
                if currentCount >= maxTokenCount {
                    let start = DispatchTime.now()
                    // Add the current output to the history
                    let chat = (role: Role.bot, content: outputString)
                    let newHistory = history + [chat]
                    logger.debug("New history: \(newHistory)")
                    guard prepare(history: newHistory) else { return output.finish() }
                    logger.debug("Truncated history and continued in \(String(format: "%.2f", Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000)) seconds")
                    logger.debug("New current count: \(self.currentCount)")
                }
            }
            return output.finish()
        } }
    }

    private(set) var isAvailable = true

    @InferenceActor
    public func waitUntilAvailable(timeout: DispatchTime) async throws {
        while !isAvailable {
            try await Task.sleep(nanoseconds: 1 * 1000 * 1000 * 1000) // 1 second
            if timeout < .now() { throw LLMError.timeout }
        }
    }

    @InferenceActor
    public func getCompletion(from input: String) async -> String {
        guard isAvailable else { fatalError("LLM is being used") }
        isAvailable = false
        let response = getResponse(from: [(.user, input)])
        var output = ""
        for await responseDelta in response {
            output += responseDelta
        }
        isAvailable = true
        return output
    }

    @InferenceActor
    public func respond(to history: [Chat], with makeOutputFrom: @escaping (AsyncStream<String>) async -> String) async {
        guard isAvailable else { return }
        isAvailable = false

        let response = getResponse(from: history)
        let output = await makeOutputFrom(response)

        postProcess(output)
        isAvailable = true
    }

    open func respond(to history: [Chat]) async -> String {
        var output = ""

        // Using a custom asynchronous closure-based method
        await respond(to: history) { [self] response in
            for await responseDelta in response {
                update(responseDelta)
                output += responseDelta
            }
            update(nil)
            return output
        }

        let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmedOutput.isEmpty ? "..." : trimmedOutput
    }

    private var multibyteCharacter: [CUnsignedChar] = []
    public func decode(_ token: Token) -> String {
        model.decode(token, with: &multibyteCharacter)
    }

    // @inlinable
    public func encode(_ text: borrowing String) -> [Token] {
        model.encode(text)
    }
}

extension Model {
    public var endToken: Token { llama_token_eos(self) }
    public var newLineToken: Token { llama_token_nl(self) }

    public func shouldAddBOS() -> Bool {
        let addBOS = llama_add_bos_token(self)
        guard addBOS != -1 else {
            return llama_vocab_type(self) == LLAMA_VOCAB_TYPE_SPM
        }
        return addBOS != 0
    }

    func decodeOnly(_ token: Token) -> String {
        var nothing: [CUnsignedChar] = []
        return decode(token, with: &nothing)
    }

    func decode(_ token: Token, with multibyteCharacter: inout [CUnsignedChar]) -> String {
        var bufferLength = 16
        var buffer: [CChar] = .init(repeating: 0, count: bufferLength)
        let actualLength = Int(llama_token_to_piece(self, token, &buffer, Int32(bufferLength)))
        guard actualLength != 0 else { return "" }
        if actualLength < 0 {
            bufferLength = -actualLength
            buffer = .init(repeating: 0, count: bufferLength)
            llama_token_to_piece(self, token, &buffer, Int32(bufferLength))
        } else {
            buffer.removeLast(bufferLength - actualLength)
        }
        if multibyteCharacter.isEmpty, let decoded = String(cString: buffer + [0], encoding: .utf8) {
            return decoded
        }
        multibyteCharacter.append(contentsOf: buffer.map { CUnsignedChar(bitPattern: $0) })
        guard let decoded = String(data: .init(multibyteCharacter), encoding: .utf8) else { return "" }
        multibyteCharacter.removeAll(keepingCapacity: true)
        return decoded
    }

    func encode(_ text: borrowing String) -> [Token] {
        let addBOS = true
        let count = Int32(text.cString(using: .utf8)!.count)
        var tokenCount = count + 1
        let cTokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(tokenCount)); defer { cTokens.deallocate() }
        tokenCount = llama_tokenize(self, text, count, cTokens, tokenCount, addBOS, false)
        let tokens = (0 ..< Int(tokenCount)).map { cTokens[$0] }
        return tokens
    }
}

private class Context {
    let pointer: OpaquePointer
    init(_ model: Model, _ params: llama_context_params) {
        pointer = llama_new_context_with_model(model, params)
    }

    deinit {
        llama_free(pointer)
    }

    func decode(_ batch: llama_batch) {
        guard llama_decode(pointer, batch) == 0 else { fatalError("llama_decode failed") }
    }
}

extension llama_batch {
    mutating func clear() {
        n_tokens = 0
    }

    mutating func add(_ token: Token, _ position: Int32, _ ids: [Int], _ logit: Bool) {
        let i = Int(n_tokens)
        self.token[i] = token
        pos[i] = position
        n_seq_id[i] = Int32(ids.count)
        if let seq_id = seq_id[i] {
            for (j, id) in ids.enumerated() {
                seq_id[j] = Int32(id)
            }
        }
        logits[i] = logit ? 1 : 0
        n_tokens += 1
    }
}

extension Token {
    enum Kind {
        case end
        case couldBeEnd
        case normal
    }
}

public enum Role {
    case user
    case bot
}

public enum LLMError: Error {
    case timeout
}

public struct Template {
    public typealias Attachment = (prefix: String, suffix: String)
    public let system: Attachment
    public let user: Attachment
    public let bot: Attachment
    public let systemPrompt: String?
    public let stopSequences: [String]
    public let prefix: String
    public let shouldDropLast: Bool

    public init(
        prefix: String = "",
        system: Attachment? = nil,
        user: Attachment? = nil,
        bot: Attachment? = nil,
        stopSequences: [String] = [],
        systemPrompt: String?,
        shouldDropLast: Bool = false
    ) {
        self.system = system ?? ("", "")
        self.user = user ?? ("", "")
        self.bot = bot ?? ("", "")
        self.stopSequences = stopSequences
        self.systemPrompt = systemPrompt
        self.prefix = prefix
        self.shouldDropLast = shouldDropLast
    }

    public var preProcess: (_ history: [Chat]) -> String {
        { [self] history in
            var processed = prefix
            if let systemPrompt {
                processed += "\(system.prefix)\(systemPrompt)\(system.suffix)"
            }
            for chat in history {
                if chat.role == .user {
                    processed += "\(user.prefix)\(chat.content)\(user.suffix)"
                } else {
                    processed += "\(bot.prefix)\(chat.content)\(bot.suffix)"
                }
            }
            if shouldDropLast {
                processed += bot.prefix.dropLast()
            } else {
                processed += bot.prefix
            }
            return processed
        }
    }

    public static func chatML(_ systemPrompt: String? = nil) -> Template {
        Template(
            system: ("<|im_start|>system\n", "<|im_end|>\n"),
            user: ("<|im_start|>user\n", "<|im_end|>\n"),
            bot: ("<|im_start|>assistant\n", "<|im_end|>\n"),
            stopSequences: ["<|im_end|>", "<|im_start|>"],
            systemPrompt: systemPrompt
        )
    }

    public static func alpaca(_ systemPrompt: String? = nil) -> Template {
        Template(
            system: ("", "\n\n"),
            user: ("### Instruction:\n", "\n\n"),
            bot: ("### Response:\n", "\n\n"),
            stopSequences: ["###"],
            systemPrompt: systemPrompt
        )
    }

    public static func llama(_ systemPrompt: String? = nil) -> Template {
        Template(
            prefix: "[INST] ",
            system: ("<<SYS>>\n", "\n<</SYS>>\n\n"),
            user: ("", " [/INST]"),
            bot: (" ", "</s><s>[INST] "),
            stopSequences: ["</s>"],
            systemPrompt: systemPrompt,
            shouldDropLast: true
        )
    }

    public static let mistral = Template(
        user: ("[INST] ", " [/INST]"),
        bot: ("", "</s> "),
        stopSequences: ["</s>", "[INST]", "[/INST]"],
        systemPrompt: nil
    )
}

extension Template: CustomStringConvertible {
    public var description: String {
        """
        Template(
            prefix: "\(prefix)",
            system: ("\(system.prefix)", "\(system.suffix)"),
            user: ("\(user.prefix)", "\(user.suffix)"),
            bot: ("\(bot.prefix)", "\(bot.suffix)"),
            stopSequences: "\(stopSequences)",
            systemPrompt: "\(systemPrompt ?? "")",
            shouldDropLast: \(shouldDropLast)
        )
        """
    }
}

public enum Quantization: String {
    case IQ2_XXS
    case IQ2_XS
    case Q2_K_S
    case Q2_K
    case Q3_K_S
    case Q3_K_M
    case Q3_K_L
    case Q4_K_S
    case Q4_K_M
    case Q5_K_S
    case Q5_K_M
    case Q6_K
    case Q8_0
}

public enum HuggingFaceError: Error {
    case network(statusCode: Int)
    case noFilteredURL
    case urlIsNilForSomeReason
}

public struct HuggingFaceModel {
    public let name: String
    public let template: Template
    public let filterRegexPattern: String

    public init(_ name: String, template: Template, filterRegexPattern: String) {
        self.name = name
        self.template = template
        self.filterRegexPattern = filterRegexPattern
    }

    public init(_ name: String, _ quantization: Quantization = .Q4_K_M, template: Template) {
        self.name = name
        self.template = template
        self.filterRegexPattern = "(?i)\(quantization.rawValue)"
    }

    package func getDownloadURLStrings() async throws -> [String] {
        let url = URL(string: "https://huggingface.co/\(name)/tree/main")!
        let data = try await url.getData()
        let content = String(data: data, encoding: .utf8)!
        let downloadURLPattern = #"(?<=href=").*\.gguf\?download=true"#
        let matches = try! downloadURLPattern.matches(in: content)
        let root = "https://huggingface.co"
        return matches.map { match in root + match }
    }

    package func getDownloadURL() async throws -> URL? {
        let urlStrings = try await getDownloadURLStrings()
        for urlString in urlStrings {
            let found = try filterRegexPattern.hasMatch(in: urlString)
            if found { return URL(string: urlString)! }
        }
        return nil
    }

    public func download(to directory: URL = .documentsDirectory, as name: String? = nil, _ updateProgress: @escaping (Double) -> Void) async throws -> URL {
        var destination: URL
        if let name {
            destination = directory.appending(path: name)
            guard !destination.exists else { updateProgress(1); return destination }
        }
        guard let downloadURL = try await getDownloadURL() else { throw HuggingFaceError.noFilteredURL }
        destination = directory.appending(path: downloadURL.lastPathComponent)
        guard !destination.exists else { return destination }
        try await downloadURL.downloadData(to: destination, updateProgress)
        return destination
    }

    public static func tinyLLaMA(_ quantization: Quantization = .Q4_K_M, _ systemPrompt: String) -> HuggingFaceModel {
        HuggingFaceModel("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", quantization, template: .chatML(systemPrompt))
    }
}

extension URL {
    @backDeployed(before: iOS 16)
    public func appending(path: String) -> URL {
        appendingPathComponent(path)
    }

    @backDeployed(before: iOS 16)
    public static var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    fileprivate var exists: Bool { FileManager.default.fileExists(atPath: path) }
    fileprivate func getData() async throws -> Data {
        let (data, response) = try await URLSession.shared.data(from: self)
        let statusCode = (response as! HTTPURLResponse).statusCode
        guard statusCode / 100 == 2 else { throw HuggingFaceError.network(statusCode: statusCode) }
        return data
    }

    fileprivate func downloadData(to destination: URL, _ updateProgress: @escaping (Double) -> Void) async throws {
        var observation: NSKeyValueObservation!
        let url: URL = try await withCheckedThrowingContinuation { continuation in
            let task = URLSession.shared.downloadTask(with: self) { url, response, error in
                if let error { return continuation.resume(throwing: error) }
                guard let url else { return continuation.resume(throwing: HuggingFaceError.urlIsNilForSomeReason) }
                let statusCode = (response as! HTTPURLResponse).statusCode
                guard statusCode / 100 == 2 else { return continuation.resume(throwing: HuggingFaceError.network(statusCode: statusCode)) }
                continuation.resume(returning: url)
            }
            observation = task.progress.observe(\.fractionCompleted) { progress, _ in
                updateProgress(progress.fractionCompleted)
            }
            task.resume()
        }
        _ = observation
        try FileManager.default.moveItem(at: url, to: destination)
    }
}

package extension String {
    func matches(in content: String) throws -> [String] {
        let pattern = try NSRegularExpression(pattern: self)
        let range = NSRange(location: 0, length: content.utf16.count)
        let matches = pattern.matches(in: content, range: range)
        return matches.map { match in String(content[Range(match.range, in: content)!]) }
    }

    func hasMatch(in content: String) throws -> Bool {
        let pattern = try NSRegularExpression(pattern: self)
        let range = NSRange(location: 0, length: content.utf16.count)
        return pattern.firstMatch(in: content, range: range) != nil
    }

    func firstMatch(in content: String) throws -> String? {
        let pattern = try NSRegularExpression(pattern: self)
        let range = NSRange(location: 0, length: content.utf16.count)
        guard let match = pattern.firstMatch(in: content, range: range) else { return nil }
        return String(content[Range(match.range, in: content)!])
    }
}

extension llama_context_params: CustomStringConvertible {
    public var description: String {
        """
                llama_context_params(
                seed: \(seed),
                n_ctx: \(n_ctx),
                n_batch: \(n_batch),
                n_threads: \(n_threads),
                n_threads_batch: \(n_threads_batch),
                embedding: \(embedding),
        """
    }
}
