BEGIN= '\033['
CLEAR = '\033[0m'
SPACE = ''

# mode methods
mode_default = '0'
mode_highlight = '1'
mode_nobold = '22'
mode_underline = '4'
mode_nounderline = '24'
mode_blink = '5'
mode_noblink = '25'
mode_reverse = '7'
mode_noreverse = '27'

# foreground color
black = '30'
red = '31'
green = '32'
yellow = '33'
blue = '34'
pink = '35'
cyan = '36'
white = '37'

# background color
BLACK = '40'
RED = '41'
GREEN = '42'
YELLOW = '43'
BLUE = '44'
PINK = '45'
CYAN = '46'
WHITE = '47'



def combine(mode, foreground, background):
    return BEGIN + ';'.join([i for i in [mode, foreground, background] if i != '']) + 'm'

def cmd_print(msg_type, msg, show = True):
    format_expression = '[{0:^11}] '
    if msg_type == 0:#INFO
        cmd_msg = combine(mode_default, blue, SPACE)+format_expression.format('INFO')+msg+CLEAR
    elif msg_type == 1:
        cmd_msg = combine(mode_highlight, red, YELLOW)+format_expression.format('WARNING')+CLEAR+combine(mode_blink, red, SPACE)+msg
    elif msg_type == 2:
        cmd_msg = combine(mode_default, red, SPACE)+format_expression.format('ERROR')+msg+CLEAR
    elif msg_type == 3:
        cmd_msg = combine(mode_default, green, SPACE)+format_expression.format('SUCCESS')+msg+CLEAR
    elif msg_type == 4:
        cmd_msg = combine(mode_default, yellow, SPACE)+format_expression.format('IMPORTANT')+msg+CLEAR
    elif msg_type == 5:
        cmd_msg = combine(mode_default, yellow, SPACE)+msg+CLEAR
    else:
        cmd_msg = msg
    if show == True:
        print(cmd_msg)
    return cmd_msg
