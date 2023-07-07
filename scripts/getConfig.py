from configobj import ConfigObj


def getConfig(dir):
    ConfigDir = dir + "/config.ini"
    Config = ConfigObj(ConfigDir)

    Config["IMGDIR"] = Config["IMGDIR"] if "IMGDIR" in Config else "img/"
    if Config["IMGDIR"][-1] not in "/\\":
        Config["IMGDIR"] += "/"
    Config["INDEX"] = int(Config["INDEX"]) if "INDEX" in Config else 0

    Config.write()
    return Config


__all__ = ["getConfig"]
