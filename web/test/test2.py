from tornado.web import Application, RequestHandler, url
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import time
import tornado.ioloop

class IndexHandler(RequestHandler):

    def get(self):
        self.write("<a href='"+self.reverse_url("login")+"'>用户登录</a>")


class RegistHandler(RequestHandler):
    def initialize(self, title):
        self.title = title

    def get(self):
        self.write("注册业务处理:" + str(self.title))


class LoginHandler(RequestHandler):
    def get(self):
        self.write("用户登录页面展示")

    def post(self):
        self.write("用户登录功能处理")


if __name__ == "__main__":
    app = Application(
        [
            (r"/", IndexHandler),
            (r"/regist", RegistHandler, {"title": "会员注册"}),
            url(r"/login", LoginHandler, name="login"),
        ]
    )

    http_server = HTTPServer(app)
    http_server.bind(8888,'0.0.0.0')
    http_server.start()
    tornado.ioloop.IOLoop.instance().start()
    time.sleep(3)
    http_server.stop()
    ioloop = tornado.ioloop.IOLoop.instance()
    ioloop.add_callback(ioloop.stop)
    print( "Asked Tornado to exit")