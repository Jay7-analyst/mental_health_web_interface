import re


class ArabicPreprocessor:
    def __init__(self):
        self.html_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
        self.email_pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
        self.phone_pattern = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
        self.money_pattern = re.compile(
            r"(\d+(?:[\.,]\d+)?)\s*(دولار|ريال|درهم|دينار|جنيه|€|\$|£|ليرة)"
        )
        self.date_like_pattern = re.compile(r"\b\d{1,2}[\/\-\._]\d{1,2}[\/\-\._]\d{2,4}\b")
        self.time_like_pattern = re.compile(r"\b\d{1,2}\s*(?::|h|H|س|ساعة)\s*\d{0,2}\b")

        self.latin_digits_pattern = re.compile(r"[A-Za-z0-9]")
        self.arabic_diacritics_pattern = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
        self.tatweel_pattern = re.compile(r"ـ+")
        self.non_arabic_pattern = re.compile(r"[^\u0600-\u06FF\s]")
        self.multi_space_pattern = re.compile(r"\s+")
        self.repeated_punct_pattern = re.compile(r"([!؟?\.،,؛;:])\1+")
        self.arabic_token_pattern = re.compile(r"[ء-ي]+")
        self.punct_pattern = re.compile(r"[،؟.!,:؛\"'“”‘’()\[\]{}<>/\\\-_=+*…]+")
        self.repeated_arabic_char_pattern = re.compile(r"([\u0600-\u06FF])\1{2,}")

        self.min_token_length = 2
        self.max_repeated_chars = 2

        self.article_keep_words = {
            "الله", "الهي", "الهيه", "الرحمن", "الرحيم",
            "الان", "الانسان", "الناس", "الذي", "التي", "الذين",
            "اللاتي", "اللاتى", "اللهم", "القران", "الحديث",
        }

        raw_negation_terms = {
            "لا", "لم", "لن", "ليس", "ليست", "لست", "لسنا",
            "بدون", "بلا", "مش", "مو", "مافي", "ما في",
        }
        self.negation_terms = {
            self._simple_normalize_token(x) for x in raw_negation_terms if x
        }

        raw_discourse_bases = {
            "شكرا", "شكرًا", "الحمد", "الله", "السلام", "عليكم",
            "رحمه", "رحمة", "بركاته", "جزاكم", "جزاك", "ارجو",
            "أرجو", "افيدوني", "أفيدوني", "دكتور", "الدكتور",
            "طبيب", "الطبيب", "سؤالي", "السؤال", "تشخيصكم",
            "حالتي", "لو", "سمحتم", "خير", "الشكر", "جزيلا",
        }
        self.discourse_bases = {
            self._simple_normalize_token(x) for x in raw_discourse_bases if x
        }

        self.stopwords = self._build_stopwords()

    def _build_stopwords(self):
        general_stopwords = {
            "في", "من", "على", "إلى", "الى", "عن", "ما", "ماذا", "هذا", "هذه",
            "ذلك", "تلك", "هو", "هي", "هم", "هن", "أنا", "انا", "نحن", "كان",
            "كانت", "يكون", "يمكن", "لقد", "قد", "ثم", "أو", "او", "و", "ف", "ب",
            "ل", "ك", "مع", "عند", "بعد", "قبل", "كل", "أي", "اي", "أحد", "احد",
            "هناك", "هنا", "لكن", "ولكن", "لأن", "لان", "إن", "ان", "إذا", "اذا",
            "حتى", "حتي", "بين", "أكثر", "اكثر", "أقل", "اقل", "أيضًا", "ايضا",
            "أحيانًا", "احيانا", "ولا", "لا", "لم", "لن", "لي", "لدي", "عندي",
            "عنده", "عندها", "الذي", "التي", "الذين", "اللاتي", "اللاتى",
            "شيء", "بعض", "أمر", "أمور", "كنت", "وأنا", "وانا",
            "أنني", "انني", "أنها", "انها", "أنه", "انه", "اني", "هل",
            "عندما", "منذ", "الآن", "الان", "الا", "إلا", "حول", "حيث", "بسبب",
            "جدا", "جدًا", "كما", "تم", "علي", "عليه", "عليها",
            "اشعر", "أشعر", "واشعر", "اعاني", "أعاني", "احس", "أحس",
            "اريد", "أريد", "استطيع", "أستطيع", "اعرف", "أعرف",
            "امي", "أمي", "ابي", "أبي",
            "الي", "وفي", "وهل", "فهل", "وما", "فانا", "بان", "ام", "مره",
            "غير", "الحاله", "الحالة", "حالتي", "حالاتي", "وعدم", "كثيرا",
            "سنه", "سنة", "لمده", "مدة", "لله", "ورحمه", "ورحمة",
            "وقد", "فقد", "وكان", "وكانت", "ايضا", "ايضاً",
        }

        consultation_filler_stopwords = {
            "السلام", "عليكم", "ورحمة", "وبركاته", "جزاكم", "خيرا", "خيرًا",
            "شكرا", "شكرًا", "أرجو", "ارجو", "افيدوني", "أفيدوني", "سؤالي",
            "السؤال", "الإجابة", "الاخ", "الأخ", "الاخت", "الأخت", "حفظه",
            "حفظها", "بارك", "بكم", "نسأل", "تعالى", "وفقكم", "الله", "بسم",
            "الرحمن", "الرحيم", "وبعد", "الفاضل", "الفاضلة", "كريم", "كريمة",
            "استشارتي", "استشارة", "جواب", "سؤالك", "رسالتك", "الموقع", "ويب",
            "اسلام", "إسلام", "المكرم", "المكرمة", "الدكتور", "الدكتورة",
            "استفسار", "استفساري", "ارجوكم", "لو", "سمحتم", "جزاك", "الرجاء",
            "أرجوكم", "حضرتك", "ممكن", "ممکن", "طبيب", "الطبيب", "دكتور",
            "علما", "علماً", "اعلم", "جزيل", "الشكر", "خير", "الجزاء",
            "صاحب", "الاستشاره", "الاستشارة", "رقم", "الرسول", "صلي", "وسلم",
            "الحمد",
        }

        hierarchy_junk = {
            "الحالات", "النفسيه", "العصبيه", "العصبية",
            "عموما", "عموماً",
        }

        dataset_specific_noise = {
            "محمد", "عبد", "العليم", "عمري", "العمر", "ابلغ",
            "اصبحت", "بدات", "بدأت", "ذهبت", "افكر",
        }

        raw_stopwords = (
            general_stopwords
            | consultation_filler_stopwords
            | hierarchy_junk
            | dataset_specific_noise
        )

        stopwords = {
            self.canonicalize_token(word)
            for word in raw_stopwords
            if word and self.canonicalize_token(word)
        }

        # Same important EDA decision: preserve negation terms.
        stopwords = stopwords - self.negation_terms

        return stopwords

    def normalize_repeated_characters(self, text):
        text = str(text)

        def repl(match):
            return match.group(1) * self.max_repeated_chars

        return self.repeated_arabic_char_pattern.sub(repl, text)

    def light_structural_preprocess(self, text):
        text = str(text)
        text = self.html_pattern.sub(" ", text)
        text = self.url_pattern.sub(" URLTOKEN ", text)
        text = self.email_pattern.sub(" EMAILTOKEN ", text)
        text = self.phone_pattern.sub(" PHONETOKEN ", text)
        text = self.money_pattern.sub(" MONEYTOKEN ", text)
        text = self.date_like_pattern.sub(" DATETOKEN ", text)
        text = self.time_like_pattern.sub(" TIMETOKEN ", text)
        text = self.repeated_punct_pattern.sub(r"\1", text)
        text = self.multi_space_pattern.sub(" ", text).strip()
        return text

    def normalize_arabic(self, text):
        text = str(text)
        text = self.normalize_repeated_characters(text)

        text = self.url_pattern.sub(" ", text)
        text = self.email_pattern.sub(" ", text)
        text = self.phone_pattern.sub(" ", text)
        text = self.money_pattern.sub(" ", text)
        text = self.date_like_pattern.sub(" ", text)
        text = self.time_like_pattern.sub(" ", text)

        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)

        text = self.arabic_diacritics_pattern.sub("", text)
        text = self.tatweel_pattern.sub("", text)
        text = self.punct_pattern.sub(" ", text)
        text = self.latin_digits_pattern.sub(" ", text)
        text = self.non_arabic_pattern.sub(" ", text)
        text = self.multi_space_pattern.sub(" ", text).strip()
        return text

    def clean_arabic_text(self, text):
        text = self.light_structural_preprocess(text)
        text = self.normalize_arabic(text)
        return text

    def _simple_normalize_token(self, token):
        token = self.normalize_arabic(token)
        token = self.multi_space_pattern.sub(" ", token).strip()
        return token

    def _strip_conservative_clitics(self, token):
        tok = token

        tok = re.sub(r"^لل(?=[\u0600-\u06FF]{2,}$)", "ال", tok)

        changed = True
        while changed:
            changed = False

            if re.match(r"^[وفبكل]ال[\u0600-\u06FF]{2,}$", tok):
                candidate = tok[1:]
                if len(candidate) >= self.min_token_length:
                    tok = candidate
                    changed = True
                    continue

            if re.match(r"^[وف][\u0600-\u06FF]{3,}$", tok):
                candidate = tok[1:]
                if candidate in self.discourse_bases:
                    tok = candidate
                    changed = True
                    continue

        return tok

    def _normalize_definite_article(self, token):
        tok = token

        if tok in self.article_keep_words:
            return tok

        if tok.startswith("ال") and len(tok) >= 5:
            candidate = tok[2:]
            if len(candidate) >= self.min_token_length:
                return candidate

        return tok

    def canonicalize_token(self, token):
        tok = self._simple_normalize_token(token)

        if not tok:
            return ""

        tok = self._strip_conservative_clitics(tok)
        tok = self._normalize_definite_article(tok)

        tok = self.punct_pattern.sub("", tok)
        tok = tok.strip()

        return tok

    def tokenize_arabic(self, text):
        text = self.clean_arabic_text(text)
        raw_tokens = self.arabic_token_pattern.findall(text)

        clean_tokens = []

        for tok in raw_tokens:
            canon = self.canonicalize_token(tok)

            if not canon:
                continue
            if len(canon) < self.min_token_length:
                continue
            if canon in self.stopwords:
                continue

            clean_tokens.append(canon)

        return clean_tokens

    def preprocess_for_model(self, patient_input):
        tokens = self.tokenize_arabic(patient_input)
        return " ".join(tokens)