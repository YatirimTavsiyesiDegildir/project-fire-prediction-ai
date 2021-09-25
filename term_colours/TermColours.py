WHITE = "\u001b[38;5;231m\033[1m"
BOLD = "\033[1m"
RED = "\u001b[38;5;196m" + BOLD
RED_BACKGROUND = "\u001b[48;5;196m" + BOLD + WHITE
GREEN = "\u001b[38;5;70m" + BOLD
GREEN_BACKGROUND = "\u001b[48;5;70m" + BOLD + WHITE
ORANGE = "\u001b[38;5;208m" + BOLD
ORANGE_BACKGROUND = "\u001b[48;5;208m" + BOLD + WHITE
GRAY = "\u001b[38;5;8m" + BOLD
GRAY_BACKGROUND = "\u001b[48;5;8m" + BOLD + WHITE
ENDCHAR = "\u001b[0m"
OK = f"{WHITE}[ {GREEN}OK {WHITE}]{ENDCHAR}"
FAIL = f"{WHITE}[ {RED}FAIL {WHITE}]{ENDCHAR}"
INFO = f"{WHITE}[ {ORANGE}INFO {WHITE}]{ENDCHAR}"
UP = f"{WHITE}[ {GREEN} UP  {WHITE}]{ENDCHAR}"
DOWN = f"{WHITE}[ {RED}DOWN {WHITE}]{ENDCHAR}"
FATAL = f"{RED_BACKGROUND}[ FATAL ]{ENDCHAR}"
WARN = f"{WHITE}[ {RED}WARN {WHITE}]{ENDCHAR}"

def info(message):
    print(
        f"{ORANGE_BACKGROUND} INFO: {ENDCHAR} {WHITE}" + message + f"{ENDCHAR}")


def success(message):
    print(
        f"{GREEN_BACKGROUND} SUCCESS: {ENDCHAR} {WHITE}" + message + f"{ENDCHAR}")


def fatal(message):
    print(
        f"{RED_BACKGROUND} FATAL: {ENDCHAR} {WHITE}" + message + f"{ENDCHAR}")


def gray_background(message):
    return f"{GRAY_BACKGROUND} {message} {ENDCHAR}"

