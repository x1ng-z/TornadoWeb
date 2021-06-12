import threading
import tornado.ioloop
import tornado.web
from tornado import gen
import asyncio

class RequestHandler(tornado.web.RequestHandler):

    # @tornado.web.asynchronous
    @gen.coroutine
    def get(self, path):
        self.write("Test")
        self.finish()

class WebServer(threading.Thread):
    def run(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        application = tornado.web.Application([
            (r"/(.*)", RequestHandler)])
        application.listen(12345)
        tornado.ioloop.IOLoop.instance().start()

WebServer().start()