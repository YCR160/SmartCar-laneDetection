from configobj import ConfigObj


def getConfig(dir):
    """
    读取配置文件并返回一个配置对象
    :param dir: 配置文件所在目录
    :return: 配置对象
    """
    ConfigDir = dir + "/config.ini"
    Config = ConfigObj(ConfigDir)

    Config["IMGDIR"] = Config["IMGDIR"] if "IMGDIR" in Config else "img/"
    if Config["IMGDIR"][-1] not in "/\\":
        Config["IMGDIR"] += "/"
    Config["INDEX"] = int(Config["INDEX"]) if "INDEX" in Config else 0

    Config.write()
    return Config


__all__ = ["getConfig"]
