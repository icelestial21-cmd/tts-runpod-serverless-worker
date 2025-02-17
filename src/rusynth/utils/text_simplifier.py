import re


class TextSimplifier:
    ''' Text simplifier for TTS '''
    def __init__(self):
        self.units = ['', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
        self.teens = ['десять', 'одиннадцать', 'двенадцать', 'тринадцать', 'четырнадцать', 'пятнадцать', 'шестнадцать', 'семнадцать', 'восемнадцать', 'девятнадцать']
        self.tens = ['', '', 'двадцать', 'тридцать', 'сорок', 'пятьдесят', 'шестьдесят', 'семьдесят', 'восемьдесят', 'девяносто']
        self.hundreds = ['', 'сто', 'двести', 'триста', 'четыреста', 'пятьсот', 'шестьсот', 'семьсот', 'восемьсот', 'девятьсот']
        self.cardinal_to_ordinal = {
            'один': 'первый', 'два': 'второй', 'три': 'третий', 'четыре': 'четвёртый',
            'пять': 'пятый', 'шесть': 'шестой', 'семь': 'седьмой', 'восемь': 'восьмой',
            'девять': 'девятый', 'десять': 'десятый', 'одиннадцать': 'одиннадцатый',
            'двенадцать': 'двенадцатый', 'тринадцать': 'тринадцатый', 'четырнадцать': 'четырнадцатый',
            'пятнадцать': 'пятнадцатый', 'шестнадцать': 'шестнадцатый', 'семнадцать': 'семнадцатый',
            'восемнадцать': 'восемнадцатый', 'девятнадцать': 'девятнадцатый', 'двадцать': 'двадцатый',
            'тридцать': 'тридцатый', 'сорок': 'сороковой', 'пятьдесят': 'пятидесятый',
            'шестьдесят': 'шестидесятый', 'семьдесят': 'семидесятый', 'восемьдесят': 'восьмидесятый',
            'девяносто': 'девяностый', 'сто': 'сотый', 'двести': 'двухсотый', 'триста': 'трёхсотый',
            'четыреста': 'четырёхсотый', 'пятьсот': 'пятисотый', 'шестьсот': 'шестисотый',
            'семьсот': 'семисотый', 'восемьсот': 'восьмисотый', 'девятьсот': 'девятисотый',
            'тысяча': 'тысячный', 'миллион': 'миллионный', 'миллиард': 'миллиардный'
        }
        self.transcription_map = {
            'ch': 'ч', 'sh': 'ш', 'th': 'т', 'ph': 'ф', 'kh': 'х', 'ts': 'ц',
            'ya': 'я', 'yo': 'ё', 'yu': 'ю', 'zh': 'ж',
            'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'э', 'f': 'ф', 'g': 'г',
            'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
            'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
            'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'й', 'z': 'з'
        }
        self.bracket_symbols = ['(', ')', '[', ']', '{', '}', '<', '>', '«', '»', '"', "'", '—', '“', '”', '…']

    def replace_brackets(self, text):
        for symbol in self.bracket_symbols:
            text = text.replace(symbol, '')
        return text

    def num_to_russian(self, num):
        def process_thousands(n, word, fem=False):
            if n == 0:
                return []
            if n == 1:
                return ['одна' if fem else 'один', word]
            if n == 2:
                return ['две' if fem else 'два', word + 'и']
            if n <= 4:
                return self.num_to_russian(n).split() + [word + 'и']
            return self.num_to_russian(n).split() + [word if n % 10 == 1 and n % 100 != 11 else word[:-1] + 'ов']

        if num == 0:
            return 'ноль'

        result = []

        for div, word, fem in [(1000000000, 'миллиард', False), (1000000, 'миллион', False), (1000, 'тысяча', True)]:
            if num >= div:
                result.extend(process_thousands(num // div, word, fem))
                num %= div

        if num >= 100:
            result.append(self.hundreds[num // 100])
            num %= 100

        if num >= 20:
            result.append(self.tens[num // 10])
            num %= 10
        elif num >= 10:
            result.append(self.teens[num - 10])
            num = 0

        if num > 0:
            result.append(self.units[num])

        return ' '.join(result)

    def inflect_numeral(self, num_word, following_word):
        words = num_word.split()
        last_word = words[-1]

        if following_word.startswith(('-й', '-я', '-е', '-ое', '-ая', '-ые')) or following_word in ['году', 'месте', 'этаже']:
            if last_word in self.cardinal_to_ordinal:
                words[-1] = self.cardinal_to_ordinal[last_word]
            elif last_word.endswith('ь'):
                words[-1] = last_word[:-1] + 'ый'
            elif last_word.endswith('к'):
                words[-1] = last_word + 'ий'
            else:
                words[-1] = last_word + 'ый'

            if following_word in ['году', 'месте', 'этаже']:
                words[-1] = words[-1][:-2] + 'ом'

        return ' '.join(words)

    def agree_with_noun(self, num_word, following_word):
        words = num_word.split()
        if words[-1] in ['один', 'одна']:
            if following_word.endswith(('а', 'я')):
                words[-1] = 'одна'
            else:
                words[-1] = 'один'
        elif words[-1] in ['два', 'две']:
            if following_word.endswith(('а', 'я')):
                words[-1] = 'две'
            else:
                words[-1] = 'два'
        return ' '.join(words)

    def replace_numbers(self, text):
        def replace_func(match):
            num = int(match.group())
            russian_num = self.num_to_russian(num)
            following_word = text[match.end():].split()[0] if text[match.end():] else ''

            russian_num = self.agree_with_noun(russian_num, following_word)
            russian_num = self.inflect_numeral(russian_num, following_word)

            return russian_num

        return re.sub(r'\b\d+\b', replace_func, text)

    def transcribe_word(self, word):
        for eng, rus in sorted(self.transcription_map.items(), key=lambda x: -len(x[0])):
            word = word.replace(eng, rus)
        return word

    def find_and_transcribe_english_words(self, text):
        english_words = re.findall(r'\b[a-zA-Z]+\b', text)

        transcribed_text = text
        for word in english_words:
            transcribed_word = self.transcribe_word(word.lower())
            transcribed_text = transcribed_text.replace(word, transcribed_word, 1)

        return transcribed_text

    def replace_initials(self, text):
        pattern = r'([А-Яа-яЁё])\.([А-Яа-яЁё])\.'
        replaced_text = re.sub(pattern, r'\1 \2', text)
        return replaced_text

    def add_space_after_comma(self, text):
        updated_string = re.sub(r',(?=[^\s])', ', ', text)
        return updated_string


    def __call__(
        self,
        text,
        replace_numbers=True,
        transcribe_english_words=True,
        replace_brackets=True,
        replace_initials=True,
        add_space_after_comma=True
    ):
        if transcribe_english_words:
            text = self.find_and_transcribe_english_words(text)
        if replace_numbers:
            text = self.replace_numbers(text)
        if replace_brackets:
            text = self.replace_brackets(text)
        if replace_initials:
            text = self.replace_initials(text)
        if add_space_after_comma:
            text = self.add_space_after_comma(text)
        return text
