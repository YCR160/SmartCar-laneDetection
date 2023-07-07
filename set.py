import compileall
import re
import sys
import os
import shutil
import time
from distutils.core import setup
from Cython.Build import cythonize

start_time = time.time()                                # 用于记录开始时间
curr_dir = os.path.abspath('.')                         # 当前目录
parent_path = sys.argv[1] if len(sys.argv) > 1 else ""  # 父目录
setup_file = __file__.replace('/', '\\')                # setup.py文件路径
build_dir = "build"                                     # 编译后的文件夹：build
build_tmp_dir = build_dir + "/temp"                     # 编译时的临时文件夹：build/temp

s = "# cython: language_level=3"  # 用于指定编译后的 pyd 文件支持的 python 版本


def get_py(base_path=os.path.abspath('.'), parent_path='', name="", excepts=(), copyOther=False, delC=False):
    """
    获取目录树中 .py 和 .pyx 文件的路径
    :param base_path: 目录树的根目录，用于搜索 .py 和 .pyx 文件
    :param parent_path: 当前正在搜索的目录的父目录
    :param excepts: 要从搜索中排除的文件路径列表
    :copyOther: 一个布尔标志，指示是否将非 Python 文件复制到构建目录
    :delC: 一个布尔标志，指示是否删除 Cython 生成的 .c 文件。
    :return: .py 和 .pyx 文件的迭代器
    """
    # 构造当前正在搜索的目录的完整路径
    full_path = os.path.join(base_path, parent_path, name)
    # 迭代当前目录中的文件和目录
    for filename in os.listdir(full_path):
        # 如果找到子目录，则使用子目录作为新的 parent_path 参数递归调用自身
        full_filename = os.path.join(full_path, filename)
        if os.path.isdir(full_filename) and filename != build_dir and not filename.startswith('.'):
            for f in get_py(base_path, os.path.join(parent_path, name), filename, excepts, copyOther, delC):
                # yield 关键字将该函数转换为生成器，允许迭代目录树中的 .py 和 .pyx 文件的路径，而无需一次性将所有路径加载到内存中
                yield f

        # 如果找到文件，则检查其扩展名。如果扩展名是 .c，则可选地删除文件（如果文件在一定时间后被修改）
        elif os.path.isfile(full_filename):
            ext = os.path.splitext(filename)[1]
            if ext == ".c":
                if delC and os.stat(full_filename).st_mtime > start_time:
                    os.remove(full_filename)
            # 如果文件不在 excepts 列表中，并且其扩展名是 .py 或 .pyx，则生成相对于根目录的文件路径
            elif full_filename not in excepts and os.path.splitext(filename)[1] not in ('.pyc', '.pyx'):
                if os.path.splitext(filename)[1] in ('.py', '.pyx') and not filename.startswith('__'):
                    path = os.path.join(parent_path, name, filename)
                    yield path
        else:
            pass


def select_all_init(path=os.path.abspath('.'), all_files=[]):
    """
    获取所有名为 __init__.py 的文件的路径
    :param path: 目录树的根目录
    :param all_files: 用于存储 __init__.py 文件的路径
    :return: __init__.py 文件的路径列表
    """
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入 list，是文件夹的话，递归
    for file in file_list:
        # 取得完整路径，并存入 cur_path 变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            select_all_init(cur_path, all_files)
        # 如果是文件，检查文件名是否为 __init__.py，如果是，则将其路径添加到列表中
        elif re.search(r"^__init__.py$", file):
            all_files.append(cur_path)
    return all_files


def pack_pyc():
    """
    将所有名为 __init__.py 的文件编译为 .pyc 文件并移动到构建目录
    :return: None
    """
    # 获取当前工作目录的绝对路径和构建目录的路径
    base_path_name = os.path.abspath(".")
    build_path_name = os.path.join(base_path_name, "build")
    assert os.path.isdir(build_path_name), "dir{}".format(build_path_name)
    # 获取当前目录及其子目录中所有名为 __init__.py 的文件的路径
    init = select_all_init()
    for file in init:
        # 将这些文件编译为 .pyc 文件并移动到构建目录
        compileall.compile_file(file, legacy=True)
        pyc_path = os.path.join(os.path.dirname(file), "__init__.pyc")
        if os.path.exists(pyc_path):
            dir_path = build_path_name + \
                os.path.dirname(file.replace(base_path_name, ''))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            shutil.copyfile(pyc_path, os.path.join(dir_path, "__init__.pyc"))
        os.remove(pyc_path)

    # 编译 start_pyd.py 文件并移动到构建目录
    start_pyd_path = os.path.join(base_path_name, "start_pyd.py")
    assert os.path.exists(
        start_pyd_path), "file not exists:{}".format(start_pyd_path)
    compileall.compile_file(start_pyd_path, legacy=True)
    start_pyd_pyc = os.path.join(base_path_name, "start_pyd.pyc")
    assert os.path.exists(
        start_pyd_pyc), "file not exists:{}".format(start_pyd_pyc)
    shutil.copyfile(start_pyd_pyc, os.path.join(
        build_path_name, "start_pyd.pyc"))
    os.remove(start_pyd_pyc)


def pack_pyd():
    """
    将指定目录及其子目录中的所有 .pyx 和 .py 文件编译为 .pyd 文件，并将它们复制到构建目录
    :return: None
    """
    # 获取指定目录及其子目录中所有 .pyx 和 .py 文件的路径
    module_list = list(
        get_py(base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,)))
    try:
        # 编译为 .pyd 文件
        setup(
            ext_modules=cythonize(module_list, compiler_directives={
                                  'language_level': 3}),
            script_args=["build_ext", "-b", build_dir, "-t", build_tmp_dir],
        )
    except Exception as ex:
        print("error! ", str(ex))
    else:
        # 将这些文件复制到构建目录
        module_list = list(get_py(base_path=curr_dir, parent_path=parent_path, excepts=(
            setup_file,), copyOther=True))

    # 删除指定目录及其子目录中所有 .c 文件
    module_list = list(get_py(
        base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,), delC=True))
    if os.path.exists(build_tmp_dir):
        shutil.rmtree(build_tmp_dir)

    print("complate! time:", time.time() - start_time, 's')


def delete_c(path='.', excepts=(setup_file,)):
    """
    删除指定目录及其子目录中编译过程中生成的 .c 文件
    :param path: 要搜索的目录的路径，默认为当前工作目录的绝对路径
    :param excepts: 要排除的文件或目录的名称
    :return: None
    """
    # 遍历指定目录中的所有文件和文件夹
    dirs = os.listdir(path)
    for dir in dirs:
        new_dir = os.path.join(path, dir)
        # 如果是文件
        if os.path.isfile(new_dir):
            ext = os.path.splitext(new_dir)[1]
            if ext == '.c':
                os.remove(new_dir)
        # 如果是文件夹，递归
        elif os.path.isdir(new_dir):
            delete_c(new_dir)


def cp_bat():
    """
    将 start_app.bat 文件复制到构建目录
    :return: None
    """
    try:
        bat = os.path.join(curr_dir, "start_app.bat")
        build = os.path.join(curr_dir, build_dir)
        shutil.copy(bat, build)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    try:
        print("----开始编译pyd文件----")
        pack_pyd()
        print("----结束编译pyd文件----")
        print("----开始编译pyc文件----")
        pack_pyc()
        print("----结束编译pyc文件----")
        print("----开始拷贝bat文件----")
        cp_bat()
        print("----结束拷贝bat文件----")
    except Exception as e:
        print(str(e))
    finally:
        delete_c()
