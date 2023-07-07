# -*- encoding: utf-8 -*-
"""
@File    : setup_bak.py
@Time    : 2020/4/21 17:55
@Author  : gaozenghui
@Site   : 
@Software: PyCharm
"""

import compileall
import re
import sys, os, shutil, time
from distutils.core import setup
from Cython.Build import cythonize

start_time = time.time()
curr_dir = os.path.abspath('.')
parent_path = sys.argv[1] if len(sys.argv) > 1 else ""
setup_file = __file__.replace('/', '\\')
build_dir = "build"
build_tmp_dir = build_dir + "/temp"

s = "# cython: language_level=3"


def get_py(base_path=os.path.abspath('.'), parent_path='', name="", excepts=(), copyOther=False, delC=False):
    """
    获取py文件的路径
    :param base_path: 根路径
    :param parent_path: 父路径
    :param excepts: 排除文件
    :return: py文件的迭代器
    """
    full_path = os.path.join(base_path, parent_path, name)
    for filename in os.listdir(full_path):
        full_filename = os.path.join(full_path, filename)
        if os.path.isdir(full_filename) and filename != build_dir and not filename.startswith('.'):
            for f in get_py(base_path, os.path.join(parent_path, name), filename, excepts, copyOther, delC):
                yield f
        elif os.path.isfile(full_filename):
            ext = os.path.splitext(filename)[1]
            if ext == ".c":
                if delC and os.stat(full_filename).st_mtime > start_time:
                    os.remove(full_filename)
            elif full_filename not in excepts and os.path.splitext(filename)[1] not in ('.pyc', '.pyx'):
                if os.path.splitext(filename)[1] in ('.py', '.pyx') and not filename.startswith('__'):
                    path = os.path.join(parent_path, name, filename)
                    yield path
        else:
            pass


def select_all_init(path=os.path.abspath('.'), all_files=[]):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            select_all_init(cur_path, all_files)
        else:
            if re.search(r"^__init__.py$", file):
                all_files.append(cur_path)
    return all_files

def pack_pyc():
    #查找所有的init文件
    base_path_name = os.path.abspath(".")
    build_path_name = os.path.join(base_path_name, "build")
    assert os.path.isdir(build_path_name), "dir{}".format(build_path_name)
    init = select_all_init()
    for file in init:
        compileall.compile_file(file, legacy=True)
        pyc_path = os.path.join(os.path.dirname(file), "__init__.pyc")
        if os.path.exists(pyc_path):
            dir_path = build_path_name + os.path.dirname(file.replace(base_path_name, ''))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            shutil.copyfile(pyc_path, os.path.join(dir_path, "__init__.pyc"))
        os.remove(pyc_path)


    start_pyd_path = os.path.join(base_path_name, "start_pyd.py")
    assert os.path.exists(start_pyd_path), "file not exists:{}".format(start_pyd_path)
    compileall.compile_file(start_pyd_path, legacy=True)
    start_pyd_pyc = os.path.join(base_path_name, "start_pyd.pyc")
    assert os.path.exists(start_pyd_pyc), "file not exists:{}".format(start_pyd_pyc)
    shutil.copyfile(start_pyd_pyc, os.path.join(build_path_name, "start_pyd.pyc"))
    os.remove(start_pyd_pyc)

def pack_pyd():
    # 获取py列表
    module_list = list(get_py(base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,)))
    try:
        setup(
            ext_modules=cythonize(module_list, compiler_directives={'language_level': 3}),
            script_args=["build_ext", "-b", build_dir, "-t", build_tmp_dir],
        )
    except Exception as ex:
        print("error! ", str(ex))
    else:
        module_list = list(get_py(base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,), copyOther=True))

    module_list = list(get_py(base_path=curr_dir, parent_path=parent_path, excepts=(setup_file,), delC=True))
    if os.path.exists(build_tmp_dir):
        shutil.rmtree(build_tmp_dir)

    print("complate! time:", time.time() - start_time, 's')

def delete_c(path='.', excepts=(setup_file,)):
    '''
    删除编译过程中生成的.c文件
    :param path:
    :param excepts:
    :return:
    '''
    dirs = os.listdir(path)
    for dir in dirs:
        new_dir = os.path.join(path, dir)
        if os.path.isfile(new_dir):
            ext = os.path.splitext(new_dir)[1]
            if ext == '.c':
                os.remove(new_dir)
        elif os.path.isdir(new_dir):
            delete_c(new_dir)

def cp_bat():
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


