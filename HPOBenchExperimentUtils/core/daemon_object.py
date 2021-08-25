import threading
from time import sleep

import Pyro4
import Pyro4.errors


class DaemonObject(object):
    def __init__(self, ns_ip, ns_port, object_ip, registration_name, logger):
        # This flag indicates that the daemon is running, so the worker can be found by the scheduler.
        # The starting procedure sets this flag to True.
        # It is the looping condition for the thread. When it is set to false, then the daemon automatically stops.
        self.thread_is_running = False

        self.logger = logger

        # Where to find the nameserver:
        self.ns_ip = ns_ip
        self.ns_port = ns_port

        # The Ip address of this object. The object's daemon creates a URI so that the others can communicate with
        # this object. Therefore, we need the current ip address.
        self.object_ip = object_ip

        # This object is registered in the nameserver with this name.
        self.registration_name = registration_name
        self.uri = None
        self.logger.debug('Going to start the daemon of this object in the background.')
        self.thread = self.__start_daemon()
        self.logger.info('Registered this object in the nameserver.')
        self.logger.debug(f'The object has the uri: {str(self.uri)}')

    def __start_daemon(self):
        """
        Start the Pyro daemon. The daemon allows other objects to interact with this worker.
        Also: make it a daemon process. Then the thread gets killed when the worker is shut down
        """
        thread = threading.Thread(target=self.__run_daemon, name=f'Thread: {self.registration_name}', daemon=True)
        thread.start()

        # Give the thread some time to start
        sleep(2)
        return thread

    def __run_daemon(self):
        """
        This is the background thread that registers the object in the nameserver. Also, it starts the request loop
        to make the object able to receive requests.

        When the flag `thread_is_running` is set to False (also from outside possible), then the request loop stops
        and the daemon shuts down.

        Since this thread is a daemon thread, it automatically stops, when the main thread terminates.
        """
        self.thread_is_running = True

        try:
            with Pyro4.Daemon(host=self.object_ip, port=0) as daemon:
                self.uri = daemon.register(self)
                self.logger.debug(f'Register {self.registration_name} at {self.uri}')

                with Pyro4.locateNS(host=self.ns_ip, port=self.ns_port) as nameserver:
                    nameserver.register(self.registration_name, self.uri)

                daemon.requestLoop(loopCondition=lambda: self.thread_is_running)
                self.logger.info(f'Stopped the request loop. This object is not reachable anymore')
        except Pyro4.errors.NamingError as e:
            self.logger.error('We could not find the nameserver. Please make sure that it is running.')
            self.logger.exception(e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __del__(self):
        self.thread_is_running = False
