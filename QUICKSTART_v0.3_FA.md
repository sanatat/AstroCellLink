# راهنمای AstroCellLink v0.3.0 — Hysteresis مکانی‌ـ‌زمانی

**وضعیت پژوهشی: اکتشافی، بدون اعتبارسنجی مستقل زیستی.** این نسخه، الگوریتم اصلی astroCaST 2024 را جایگزین نمی‌کند و سلول‌های منفرد را از کانال dFF تشخیص قطعی نمی‌دهد.

## به‌روزرسانی بدون حذف داده واقعی

این ZIP با ساختار `AstroCellLink/...` عرضه می‌شود. فایل‌های `real_data`، `.venv`، پروژه‌های شخصی در ریشه و مدل‌های کاربر داخل آن **نیستند**. با این حال قبل از جایگزینی کدها، نسخه پشتیبان از پوشه‌های کد بگیرید:

```bash
cd ~/AstroCellLink
# فقط کد پروژه؛ فایل چندصد مگابایتی ExVivo در بکاپ کپی نمی‌شود
[ -d astrocelllink ] && tar -czf ~/AstroCellLink_source_before_v03.tar.gz \
    astrocelllink tests scripts pyproject.toml README.md

cd ~
unzip -o ~/Downloads/AstroCellLink_v0.3.0.zip -d ~
cd ~/AstroCellLink
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
astrocelllink hysteresis --help
```

**انتظار تست:** ۳۰ تست موفق (۲۱ قبلی + ۹ جدید)، در محیط توسعه آزمایش شده‌اند؛ نتیجه روی ماشین شما باید مستقلاً بررسی شود.

## اجرای پیش‌فرض واقعی — انتخاب پارامترها از پیش

ما از برچسب ۳۳۵ رویداد کاندید قبلی به عنوان هسته‌های قوی استفاده می‌کنیم. آستانه ضعیف z=1.0 بر پایه میانه و MAD زمانی هر پیکسل است. برای گسترش، وجود شواهد در حداقل دو فریم متوالی، اتصال سه‌بعدی و محدودیت فاصله از هسته لازم است. تنظیمات زیر **مقادیر اولیه اکتشافی** هستند و در داده برچسب‌خورده بهینه نشده‌اند.

```bash
cd ~/AstroCellLink
source .venv/bin/activate

astrocelllink hysteresis \
  --candidate real_data/exvivo_screen/candidate_event_map.tiff \
  --movie real_data/exvivo_dff_stride2.h5 \
  --dataset dff/ch0 \
  --weak-z 1.0 \
  --spatial-radius 2 \
  --temporal-radius 2 \
  --connectivity 1 \
  --min-consecutive 2 \
  --reference real_data/exvivo_upstream_v042.roi/event_map.tiff \
  --frame 180 \
  --output real_data/hysteresis_v03
```

وجود `--reference` فقط برای **محاسبه توافق بعد از پردازش** است؛ انتخاب، اتصال و برچسب‌گذاری voxels هیچ‌گاه از astroCaST استفاده نمی‌کند. برای جلوگیری از انتخاب ابرپارامترها بر اساس همان داده، پارامترها را پیش از دیدن benchmark تثبیت کنید.

برای گزارش خلاصه، مشاهده تصویر، و مقایسه قبل/بعد:

```bash
cat real_data/hysteresis_v03/hysteresis_summary.json
xdg-open real_data/hysteresis_v03/hysteresis_qc.png
cat real_data/hysteresis_v03/agreement_before/comparison.json
cat real_data/hysteresis_v03/agreement_after/comparison.json
```

## خروجی‌ها

| فایل | محتوا |
| --- | --- |
| `refined_event_map.tiff` | برچسب سه‌بعدی اصلاح‌شده؛ **شناسه‌های رویداد قبلی حفظ می‌شوند** |
| `hysteresis_summary.json` | تعداد پیکسل-فریم افزوده، تنظیمات، تعداد اتصال‌های مبهم و Jaccard قبل/بعد |
| `per_event_growth.csv` | آمار توسعه هر رویداد |
| `hysteresis_qc.png` | تصویر کنترلی فریم ۱۸۰ |
| `agreement_before/` و `agreement_after/` | توافق با astroCaST؛ تنها در صورت دادن `--reference` |

نقشه اصلی کاندیدها و پوشه `exvivo_upstream_v042.roi` تغییر داده نمی‌شوند. برای اجرای مجدد از پوشه خروجی **جدید** استفاده کنید؛ برنامه پوشه غیرخالی قبلی را بازنویسی نمی‌کند.

## محدودیت‌ها و کنترل‌های لازم

1. Hysteresis بدون هسته نمی‌تواند رویداد جدید بسازد. ممکن است برخی سیگنال‌های ضعیف واقعی همچنان از دست بروند.
2. حفظ شناسه‌ها با Watershed صرفاً یک **سیاست عددی** است، نه تضمین جدایی زیستی رویدادها. `components_with_competing_seed_ids` را حتماً بررسی کنید.
3. آستانه نرمال‌شده به MAD یک امتیاز اکتشافی است، نه آزمون معنی‌داری کالیبره‌شده.
4. Jaccard با astroCaST معیار **توافق دو detector** است، نه حساسیت و ویژگی زیستی.
5. برای مقاله، لازم است چند فایل مستقل، annotation متخصص، کنترل منفی و ارزیابی split/merge مستقل وجود داشته باشد.
6. محاسبات کل ویدئو را داخل RAM بارگذاری می‌کنند. بنچمارک مصنوعی با ابعاد `(281,251,251)` در محیط آزمایش حدود ۰٫۸ GB حداکثر RSS داشته، اما روی لپ‌تاپ شما و با مقایسه اضافی ممکن است مصرف RAM بیشتر باشد.
7. حالت `--baseline rolling_percentile` با `--rolling-window 41` برای افت یا تغییر خط پایه آزمایش‌پذیر است، ولی بسیار کندتر بوده و نباید پس از دیدن نتیجه مرجع برای نتیجه‌گیری مستقل تنظیم شود.

برای مقایسه مجدد ایمن با نسخه قبلی، با حفظ خروجی قدیمی، از دستور موجود استفاده کنید:

```bash
astrocelllink compare-events \
  --candidate real_data/hysteresis_v03/refined_event_map.tiff \
  --astrocast real_data/exvivo_upstream_v042.roi \
  --min-iou 0.10 --frame 180 \
  --output real_data/hysteresis_v03_compare_separate
```

## مرور مهندسی کد

`astrocelllink/hysteresis.py` — محاسبه score، آستانه ضعیف، استمرار زمانی، corridor فضایی/زمانی، propagation بر مبنای اتصال، watershed با حفاظت از seed-ID، صادرات QC و گزارش. تست‌های آن در `tests/test_hysteresis_v030.py` هستند. محیط `astrocast310` همچنان فقط برای اجرای نسخه اصلی astroCaST استفاده شود؛ در محیط آن، پروژه AstroCellLink را نصب نکنید.
