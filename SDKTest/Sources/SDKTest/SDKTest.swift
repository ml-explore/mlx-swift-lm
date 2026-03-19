import Foundation

@MainActor
func test() {
    
    class Event {
        let id = UUID()
        let title: String
        
        init(title: String) {
            self.title = title
        }
    }
    
    // this is macOS26 API
    struct TestNotification: NotificationCenter.MainActorMessage {
        typealias Subject = Event
    }

    let event = Event(title: "WWDC")

    // Observe notifications for a specific subject instance
    let token = NotificationCenter.default.addObserver(
        of: event,
        for: TestNotification.self
    ) { message in
        print("Received TestNotification")
    }

    // Post to all observers watching that specific event
    NotificationCenter.default.post(TestNotification(), subject: event)

    _ = token // ObservationToken — observation lives as long as this does
}
