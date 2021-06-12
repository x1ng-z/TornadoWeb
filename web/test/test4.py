import threading
import tornado.ioloop
import tornado.web
import time
from tornado.httpserver import HTTPServer


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world!\n")

def start_tornado(*args, **kwargs):
    application = tornado.web.Application([
        (r"/", MainHandler),
    ])

    http_server = HTTPServer(application)
    http_server.bind(8008, '0.0.0.0')
    http_server.start()
    print("Starting Torando")
    tornado.ioloop.IOLoop.instance().start()
    print ("Tornado finished")

def stop_tornado():
    ioloop = tornado.ioloop.IOLoop.instance()
    ioloop.add_callback(ioloop.stop)
    print ("Asked Tornado to exit")

def main():

    t = threading.Thread(target=start_tornado)
    t.start()

    time.sleep(5)

    stop_tornado()
    t.join()

if __name__ == "__main__":
    main()