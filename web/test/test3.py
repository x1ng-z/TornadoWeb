# coding: utf-8
"""
version: v2
update: 2016-11-21 by arron
comments: async handle
"""
# one job contains many tasks.
import os
import sys
import logging
import tornado
import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.gen
from tornado.concurrent import run_on_executor
from tornado.escape import json_decode
# 这个并发库在python3自带在python2需要安装sudo pip install futures
from concurrent.futures import ThreadPoolExecutor
import time
import json
import crypt

LISTEN_PORT = 8088
PROCESS_NUM = 1
TOP_PATH = "/dev/shm"
LOG_FILENAME = "{script_name}.log".format(script_name=sys.argv[0].rstrip('.py'))
SALT = 'itcac'


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        say_hi = "Hello, I am Cacti. <p>Can I help you?<p>\n"
        self.write(say_hi)


class UpdateHandler(tornado.web.RequestHandler):
    def get(self):
        self.handle()

    def post(self):
        self.handle()

    def handle(self):
        try:
            logging.info(self.request.body)
            clientIP = self.get_argument('ip')
            self.write('1')
        except Exception as e:
                logging.error("%s" % str(e))
                self.write('2')


class TaskHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(300)

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        # 假如你执行的异步会返回值被继续调用可以这样(只是为了演示),否则直接yield就行
        res = yield self.handle()
        self.write(res)
        # self.finish()
        # self.handle()

    @tornado.gen.coroutine
    def post(self):
        # 假如你执行的异步会返回值被继续调用可以这样(只是为了演示),否则直接yield就行
        res = yield self.handle()
        self.write(res)
        # self.finish()
        # self.handle()

    @run_on_executor
    def handle(self):
        resp = ''
        try:
            logging.error(self.request.body)
            data = json.loads(self.request.body)
            logging.error("%s, %s" % (type(data), data))
            token = data['token']
            result = {}
            result['userid'] = '15810923357'
            result['token'] = token
            result['cid'] = 85492
            result['a'] = 'test'
            resp = json.dumps(result)
        except Exception as e:
            logging.error("%s" % str(e) + " " + str(self.request.remote_ip))
            resp = json.dumps({"code": "2", "msg": str(e)})
        finally:
            return resp


settings = {
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
    # "cookie_secret": "61oETzKXQAGaYdkL5gEmGeJJFuYh7EQnp2XdTP1o/Vo=",
    # "login_url": "/login",
    # "xsrf_cookies": True,
}

application = tornado.web.Application([
    (r"/", MainHandler),
    (r"/md5json", TaskHandler),
    (r"/update", UpdateHandler),
], **settings)

if __name__ == "__main__":
    # reload(sys)
    sys.setdefaultencoding('utf8')

    # logging.basicConfig(filename=config.LOG_FILENAME, level=logging.INFO)
    rotate_handler = logging.handlers.TimedRotatingFileHandler(LOG_FILENAME, 'D', 1, 0)
    rotate_handler.suffix = "%Y%m%d%H"
    str_format = '%(asctime)s %(levelname)s %(module)s.%(funcName)s Line.%(lineno)d: %(message)s'
    log_format = logging.Formatter(str_format)
    rotate_handler.setFormatter(log_format)
    logging.getLogger().addHandler(rotate_handler)
    logging.getLogger().setLevel(logging.ERROR)

    # application.listen(config.LISTEN_PORT)
    # tornado.ioloop.IOLoop.instance().start()
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.bind(LISTEN_PORT)
    http_server.start(PROCESS_NUM)
    tornado.ioloop.IOLoop.instance().start()