"""The HTTP listen backlog holds a burst of simultaneous clients."""
import socket
import unittest

from serve.handler import Server


class ListenBacklog(unittest.TestCase):
    def test_backlog_is_wide(self):
        self.assertGreaterEqual(Server.request_queue_size, 128)

    def test_a_burst_of_connections_waits_in_the_queue(self):
        # nothing accepts yet: every client must still get through the handshake
        srv = Server(("127.0.0.1", 0), lambda *a: None)
        clients = []
        try:
            port = srv.server_address[1]
            for _ in range(60):
                c = socket.create_connection(("127.0.0.1", port), timeout=2)
                clients.append(c)
            self.assertEqual(len(clients), 60)
        finally:
            for c in clients:
                c.close()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
