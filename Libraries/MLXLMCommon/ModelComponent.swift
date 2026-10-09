// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A part of a model that can load separately from the rest, such as the vision encoder of a VLM.
///
/// A model declares its components through ``ModelComponentsProviding``. A component that the
/// model lists in ``ModelComponentsProviding/onDemandComponents`` loads the first time an input
/// needs it. List a component in ``ModelConfiguration/excludedComponents`` to never load it.
public struct ModelComponent: Hashable, Sendable, CustomStringConvertible {
    public let name: String

    public init(_ name: String) {
        self.name = name
    }

    public var description: String { name }

    /// Image and video encoders, with the projectors that feed them to the language model.
    public static let vision = ModelComponent("vision")
}

/// When a load reads the components that a model can load on first use.
public enum ComponentLoading: Sendable {
    /// Load each of ``ModelComponentsProviding/onDemandComponents`` the first time an input needs
    /// it, the way an `MLXArray` is computed only when it is evaluated. The checkpoint files must
    /// stay readable until then.
    case onFirstUse

    /// Load every component with the model, as model conversion needs.
    case immediate
}

/// A model with components that can load separately or not at all.
///
/// Before the loader reads the checkpoint, it removes the modules of each component that does not
/// load now, and it does not read their tensors. The module tree does not change after the load.
/// An on-demand component loads later through the same
/// ``BaseLanguageModel/prepareCheckpoint(_:)``, quantization and validation as the model.
public protocol ModelComponentsProviding: BaseLanguageModel {
    /// The checkpoint components that make up each model component.
    ///
    /// Each ``CheckpointComponent/destination`` is the path of an optional module. Its
    /// ``CheckpointComponent/namespaces`` list every serialized prefix of the module's tensors,
    /// as they appear before ``BaseLanguageModel/prepareCheckpoint(_:)``.
    var modelComponents: [ModelComponent: [CheckpointComponent]] { get }

    /// Components that load the first time an input needs them. The default is empty.
    var onDemandComponents: [AnyOnDemandComponent] { get }
}

extension ModelComponentsProviding {
    public var onDemandComponents: [AnyOnDemandComponent] { [] }
}

/// A model component that loads the first time it is used. See ``OnDemandComponent``.
///
/// It lives outside the model's module tree, so loading it does not change the model. It reads,
/// verifies and evaluates the component under a load lock, and publishes it only after that.
/// Concurrent first uses share one load. A published component does not change. State queries
/// such as ``AnyOnDemandComponent/isLoaded`` never wait for a load in progress.
public class AnyOnDemandComponent: @unchecked Sendable {
    public let component: ModelComponent

    let make: @Sendable () -> Module
    /// Guards `state`. Never held while the checkpoint is read.
    private let stateLock = NSLock()
    /// Serializes loads, so concurrent first uses share one load.
    private let loadLock = NSLock()
    private var state = State.unavailable

    private enum State {
        case unavailable
        case pending(Source)
        case loaded(Module)
    }

    init(_ component: ModelComponent, make: @escaping @Sendable () -> Module) {
        self.component = component
        self.make = make
    }

    /// Whether the component has loaded.
    public var isLoaded: Bool {
        stateLock.withLock {
            if case .loaded = state { true } else { false }
        }
    }

    /// Whether a source is set and the component has not loaded yet.
    var isPending: Bool {
        stateLock.withLock {
            if case .pending = state { true } else { false }
        }
    }

    /// Load on a GCD queue. Loading blocks on file I/O, and Swift concurrency threads must not.
    func loadInBackground() async throws {
        try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Void, any Error>) in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Result { _ = try self.loadModule() })
            }
        }
    }

    /// Where to load the component from, or `nil` if it does not load on demand.
    func setSource(_ source: Source?) {
        stateLock.withLock { state = source.map(State.pending) ?? .unavailable }
    }

    /// A failed load leaves the component pending, and the next call tries again.
    func loadModule() throws -> Module {
        try loadLock.withLock {
            switch stateLock.withLock({ state }) {
            case .loaded(let module):
                return module
            case .unavailable:
                throw OnDemandComponentError.unavailable(component)
            case .pending(let source):
                let module = make()
                try applyCheckpoint(source.checkpoint(), to: module)
                eval(module)
                stateLock.withLock { state = .loaded(module) }
                return module
            }
        }
    }

    /// The component's tensors, prepared by the model.
    struct Source {
        let urls: [URL]
        let namespaces: [String]
        let perLayerQuantization: BaseConfiguration.PerLayerQuantization?
        let prepare: (ModelCheckpoint) throws -> ModelCheckpoint

        func checkpoint() throws -> ModelCheckpoint {
            let namespaces = namespaces
            var checkpoint = try loadModelCheckpoint(urls: urls) { name in
                namespaces.contains { CheckpointNameMapping.relativeName(name, in: $0) != nil }
            }
            checkpoint.perLayerQuantization = perLayerQuantization
            return try prepare(checkpoint)
        }
    }
}

/// A model component of type `Content` that loads the first time it is used.
///
/// `Content` holds the component's modules at the paths they have in the model:
///
/// ```swift
/// private let vision = OnDemandComponent(.vision) { VisionStack(config) }
///
/// public var onDemandComponents: [AnyOnDemandComponent] { [vision] }
///
/// let stack = try vision.load()
/// ```
public final class OnDemandComponent<Content: Module>: AnyOnDemandComponent, @unchecked Sendable {
    /// - Parameters:
    ///   - component: the component this loads
    ///   - make: builds new, unloaded modules for the component
    public init(_ component: ModelComponent, make: @escaping @Sendable () -> Content) {
        super.init(component, make: make)
    }

    /// The component's modules, loaded on first use.
    public func load() throws -> Content {
        // Only `make`, which returns `Content`, creates the module.
        unsafeDowncast(try loadModule(), to: Content.self)
    }
}

public enum OnDemandComponentError: LocalizedError {
    case unavailable(ModelComponent)

    public var errorDescription: String? {
        switch self {
        case .unavailable(let component):
            "The model was loaded without its \(component) component."
        }
    }
}

extension ModelComponent {
    /// The components that `input`'s media need.
    static func needed(by input: LMInput) -> Set<ModelComponent> {
        input.image != nil || input.video != nil ? [.vision] : []
    }
}

/// The on-demand components of `model` among `needed` that have not loaded yet.
func pendingOnDemandComponents(of model: BaseLanguageModel, among needed: Set<ModelComponent>)
    -> [AnyOnDemandComponent]
{
    guard !needed.isEmpty, let model = model as? any ModelComponentsProviding else { return [] }
    return model.onDemandComponents.filter { needed.contains($0.component) && $0.isPending }
}

/// Load `components` before prefill, off the Swift concurrency thread pool.
func loadOnDemandComponents(_ components: [AnyOnDemandComponent]) async throws {
    for component in components {
        try await component.loadInBackground()
    }
}

/// Arrange `model`'s component modules for a load from `urls`, while the loader has exclusive
/// access to the model.
///
/// Returns the serialized namespaces the load must not read.
func prepareComponents(
    of model: BaseLanguageModel, urls: [URL], excluded: Set<ModelComponent>,
    loading: ComponentLoading, perLayerQuantization: BaseConfiguration.PerLayerQuantization?
) throws -> [String] {
    guard let model = model as? any ModelComponentsProviding else { return [] }
    let onDemand = Dictionary(
        model.onDemandComponents.map { ($0.component, $0) }, uniquingKeysWith: { first, _ in first }
    )
    let attached = Set(model.namedModules().map(\.0))

    var unread = [String]()
    for (component, parts) in model.modelComponents {
        let holder = onDemand[component]
        let isExcluded = excluded.contains(component)
        let deferred = holder != nil && loading == .onFirstUse && !isExcluded
        holder?.setSource(
            deferred
                ? .init(
                    urls: urls, namespaces: parts.flatMap(\.namespaces),
                    perLayerQuantization: perLayerQuantization,
                    prepare: { [weak model] checkpoint in
                        guard let model else { throw OnDemandComponentError.unavailable(component) }
                        return try model.prepareCheckpoint(checkpoint)
                    })
                : nil)

        if isExcluded || deferred {
            for part in parts where attached.contains(part.destination) {
                try model.setModule(nil, at: part.destination)
            }
            unread += parts.flatMap(\.namespaces)
        } else if let holder {
            // An earlier on-demand load removed these modules; rebuild them.
            let built = Dictionary(uniqueKeysWithValues: holder.make().namedModules())
            for part in parts where !attached.contains(part.destination) {
                try model.setModule(built[part.destination], at: part.destination)
            }
        }
    }
    return unread
}

extension Module {
    /// Set or remove the optional module at the dot-separated `path`.
    fileprivate func setModule(_ module: Module?, at path: String) throws {
        let keys = path.components(separatedBy: ".")
        var item = module.map { NestedItem<String, Module>.value($0) } ?? .none
        for key in keys.dropFirst().reversed() {
            item = .dictionary([key: item])
        }
        try update(modules: ModuleChildren(values: [keys[0]: item]), verify: .all)
    }
}
