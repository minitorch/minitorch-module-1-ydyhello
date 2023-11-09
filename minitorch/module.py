class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        training (bool): Whether the module is in training mode or evaluation mode

    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self):
        "Return the direct child modules of this module."
        return self.__dict__["_modules"].values()

    def train(self):
        # 将当前模块及其所有子模块切换到训练模式，以便在模型训练时使用
        "Set the mode of this module and all descendent modules to `train`."
        # TODO: Implement for Task 0.4.
        self.training = True # 切换到训练模式
        for child_ in self._modules:
            self._modules[child_].train() # 对每个子模块调用 train 方法
        # raise NotImplementedError('Need to implement for Task 0.4')

    def eval(self):
        # 评估模型
        "Set the mode of this module and all descendent modules to `eval`."
        # TODO: Implement for Task 0.4.
        self.training = False
        for child_ in self._modules:
            self._modules[child_].eval()

    def named_parameters(self): 
        # 收集当前模块及其所有子模块的参数
        # 并以层次结构的方式返回参数的名称和对应的 Parameter 对象
        """
        Collect all the parameters of this module and its descendents.

        Returns:
            list of pairs: Contains the name and :class:`Parameter` of each ancestor parameter.
        """
        # TODO: Implement for Task 0.4.
        res = [] # 存储参数的名称和 Parameter 对象
        for k, v in self._parameters.items():
            res.append((k, v)) # 参数的名称 k 和 Parameter 对象 v 添加到列表 res 中

        for child_ in self._modules:
            child_params = self._modules[child_].named_parameters() # 获取子模块及其子模块的参数列表
            for item in child_params: # 将子模块的参数添加到 res 列表中
                res.append((child_ + '.' + item[0], item[1]))

        return res

        # raise NotImplementedError('Need to implement for Task 0.4')

    def parameters(self):
        # 遍历当前模块及其所有子模块的参数，并以列表形式返回这些参数
        "Enumerate over all the parameters of this module and its descendents."
        # TODO: Implement for Task 0.4.
        # 初始化为当前模块的 _parameters 字典中所有参数的值（Parameter 对象）构成的列表
        res = list(self._parameters.values())

        for child_ in self._modules:
            child_params = self._modules[child_].parameters()
            for item in child_params:
                res.append(item)
        return res

    def add_parameter(self, k, v):
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k (str): Local name of the parameter.
            v (value): Value for the parameter.

        Returns:
            Parameter: Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        assert False, "Not Implemented"

    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x=None, name=None):
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x):
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)