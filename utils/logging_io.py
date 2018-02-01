BEGIN= '\033['
CLEAR = '\033[0m'
SPACE = ''
NONE = ''

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

logging_content = []

def combine(mode, foreground, background):
    return ';'.join([i for i in [mode, foreground, background] if i != '']) + 'm'

def wrapper(colors, content):
    return BEGIN + '{}{}'.format(colors, content) + CLEAR

class logging_io(object):

    def __init__(self):

        pass

    @staticmethod
    def DEBUG_INFO(content):
        content_ = '{0:10}:{1}'.format('[DEBUG]', content)
        logging_content.append(content_)
        print(wrapper(combine(mode_default, green, BLACK), content_))

    @staticmethod
    def DEFAULT_INFO(content):
        content_ = '{0:10} {1}'.format(' ', content)
        logging_content.append(content_)
        print(content_)

    @staticmethod
    def WARNING_INFO(content):
        content_ = '{0:10}:{1}'.format('[WARNING]', content)
        logging_content.append(content_)
        print(wrapper(combine(mode_blink, red, YELLOW), content_))

    @staticmethod
    def ERROR_INFO(content):
        content_ = '{0:10}:{1}'.format('[ERROR]', content)
        logging_content.append(content_)
        print(wrapper(combine(mode_default, red, NONE), content_))

    @staticmethod
    def SUCCESS_INFO(content):
        content_ = '{0:10}:{1}'.format('[SUCCESS]', content)
        logging_content.append(content_)
        print(wrapper(combine(mode_default, blue, NONE), content_))

    @staticmethod
    def BUILD_INFO(content):
        content_ = '{0:10}:{1}'.format('[BUILD]', content)
        logging_content.append(content_)
        print(wrapper(combine(mode_default, pink, NONE), content_))

    @staticmethod
    def RESULT_INFO(content):
        content_ = '{0:10}:{1}'.format('[RESULT]', content)
        logging_content.append(content_)
        print(wrapper(combine(mode_default, yellow, NONE), content_))

    @staticmethod
    def LOG_COLLECTOR(filename):
        writer = open(filename, 'w', encoding='utf-8')
        for line in logging_content:
            writer.write(line+'\n')
        writer.close()




if __name__ == '__main__':
    logging_io.DEBUG_INFO('This is a DEBUG_INFO demo.')
    logging_io.DEFAULT_INFO('This is a DEFAULT_INFO demo.')
    logging_io.WARNING_INFO('This is a WARNING_INFO demo.')
    logging_io.ERROR_INFO('This is a ERROR_INFO demo.')
    logging_io.SUCCESS_INFO('This is a SUCCESS_INFO demo.')
    logging_io.BUILD_INFO('This is a BUILD_INFO demo.')
    logging_io.RESULT_INFO('This is a RESULT_INFO demo.')
    logging_io.LOG_COLLECTOR('log.txt')
