# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    '''
    Registry是一个用来管理注册模块的helper类，它继承了一个字典类，并且添加了register方法。
    举一个如何使用的例子：
    1、创建一个registry对象
        some_registry = Registry()    (有没有参数都可以)
    2、注册新的模块 （有两种方式）
       方式一：正常的方式是调用register函数进行注册：
             # 首先定义一个模块 foo
             def foo():
                 ...
             # 然后在registry对象上注册它
             # foo_module相当于字典的key，foo相当于字典的value(简单吧 hhh)
             some_registry.register("foo_module", foo)
       方式二：在声明某个模块时，作为一个装饰器使用：
             @some_registry.register("foo_module")
             @some_registry.register("foo_module_nickname")
              def foo():
                  ...
             # 效果等同于  ：some_registry.register("foo_module", foo)
             #              some_registry.register("foo_module_nickname", foo)
     3、获取被注册的模块(像使用字典一样)
          f = some_registry["foo_module"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    # 就是一个装饰器
    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            # 该函数是将module_name 和 moudle 以key 和 value 的形式加到字典当中去
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn
