import time
import signal
import logging
import os
import sys

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import threading

import asyncio
from tornado.options import define, options

define("port", default=8888, help="run on the given port", type=int)
MAX_WAIT_SECONDS_BEFORE_SHUTDOWN = 3

class ProxyServer:

    def __init__(self):
        self.__app = None
        self.__server = None
        self.ioloop = tornado.ioloop.IOLoop.instance()

    def __call__(self):
        tornado.options.parse_command_line()
        self.__app = tornado.web.Application([
                        (r"/", MainHandler),
                    ])
        self.__server = tornado.httpserver.HTTPServer(self.__app, xheaders=True)
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.__server.listen(options.port)
        pid = os.getpid()
        print("ProxyServer pid=%d" % pid)
        self.ioloop.start()

    def sig_handler(self, sig, frame):
        logging.warning('Caught signal: %s', sig)
        self.ioloop.add_callback(self.shutdown)

    def shutdown(self):
        logging.info('Stopping http server')
        self.__server.stop()

        logging.info('Will shutdown in %s seconds ...', MAX_WAIT_SECONDS_BEFORE_SHUTDOWN)

        deadline = time.time() + MAX_WAIT_SECONDS_BEFORE_SHUTDOWN

        def stop_loop():
            now = time.time()
            if now < deadline and (self.ioloop._callbacks or self.ioloop._timeouts):
                self.ioloop.add_timeout(now + 1, stop_loop)
            else:
                self.ioloop.stop()
                logging.info('Shutdown')
        stop_loop()

    @staticmethod
    def stop():
        pid = os.getpid()
        print("stop pid=%d" % pid)
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGINT)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


def main():
    count = 0
    while count != 10:
        proxy = ProxyServer()
        th = threading.Thread(target=proxy)
        th.start()
        signal.signal(signal.SIGTERM, proxy.sig_handler)
        signal.signal(signal.SIGINT, proxy.sig_handler)
        print("In main {}".format(count))
        time.sleep(10)
        ProxyServer.stop()
        time.sleep(5)
        count += 1

if __name__ == "__main__":
    pid = os.getpid()
    print("main pid=%d" % pid)
    main()