import logging

from netifaces import ifaddresses, AF_INET, interfaces

_log = logging.getLogger(__name__)


def nic_name_to_host(nic_name):
    """ Helper function to translate the name of a network card into a valid host name"""
    valid_interfaces = interfaces()
    assert nic_name in valid_interfaces, f'The given interface ({nic_name}) is not in the list of valid interfaces: ' \
                                         f'{", ".join(valid_interfaces)}'
    host = ifaddresses(nic_name).setdefault(AF_INET, [{'addr': 'No IP addr'}])[0]
    host = host['addr']
    _log.debug(f'Interface {nic_name} generates the IP: {host}')
    return host
