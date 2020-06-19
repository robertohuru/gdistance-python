# encoding: utf-8
# module gdistance.core
# from core.cp37-win_amd64.pyd
# by generator 1.147
""" Provides some functions, but faster """
# no imports

# functions
def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, imp
    __file__ = pkg_resources.resource_filename(__name__, 'core.cp37-win_amd64.pyd')
    __loader__ = None; del __bootstrap__, __loader__
    imp.load_dynamic(__name__,__file__)
__bootstrap__()