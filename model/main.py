from cortex_client.webserver import webserver_app
from cortex_client import ModelRouter
from cortex_client import ModelRunner

from model import RealestateRF


webserver_app.modelrunner = ModelRunner(RealestateRF())

if __name__ == '__main__':

    ModelRouter.main(RealestateRF())
