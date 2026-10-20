# راهنمای عملی AstroCellLink 0.2 روی Linux Mint

**هدف:** مقایسه آشکارساز آزمایشی ما و astroCaST 0.4.2 روی فیلم واقعی `ExVivoSuppRaw.h5`، سپس غربالگری/ارزیابی ماسک‌های سلولی.

**محدودیت:** هیچ‌یک از نتایج بدون ماسک‌های مرجع مستقل «تشخیص صحیح هر آستروسیت» یا «موج کلسیمی اثبات‌شده» نیست.

## گام ۰: حفظ داده‌های موجود و به‌روزرسانی کد

فایل ZIP جدید را به `~/Downloads` منتقل کن. پروژه قدیمی و `real_data` روی سیستم تو باقی می‌مانند؛ از کد کنونی نسخه پشتیبان بگیر:

```bash
cd ~/AstroCellLink
cp -a astrocelllink "astrocelllink_backup_before_v02"
cd ~
unzip -o ~/Downloads/AstroCellLink_v0.2.0.zip -d ~
cd ~/AstroCellLink
source .venv/bin/activate
python -m pip install -e '.[test]'
astrocelllink --help
pytest -q
```

**نکته:** مسیر داده `~/AstroCellLink/real_data` نباید پاک یا جابه‌جا شود. هرگز `rm -rf ~/AstroCellLink` اجرا نکن.

## گام ۱: داده هم‌رزولوشن بساز (محیط فعلی Python 3.12)

نقشه رویدادهای کاندید موجود از ۲۸۱ فریم ۲۵۱×۲۵۱ با stride=2 ساخته شده است؛ آشکارساز رسمی را روی همان داده زیر اجرا خواهیم کرد:

```bash
cd ~/AstroCellLink
source .venv/bin/activate
astrocelllink prepare-h5 \
  --input real_data/public_data/ExVivoSuppRaw.h5 \
  --dataset dff/ch0 \
  --stride 2 \
  --output real_data/exvivo_dff_stride2.h5
```

این فایل ممکن است ده‌ها مگابایت باشد و در یک فایل تازه ساخته می‌شود. اگر وجود داشته باشد، دستور برای جلوگیری از بازنویسی متوقف می‌شود.

## گام ۲: محیط جداگانه astroCaST رسمی Python 3.9

astroCaST 0.4.2 طبق `pyproject.toml` رسمی، `Python >=3.9,<3.11` را نیاز دارد؛ آن را داخل `.venv` مربوط به Python 3.12 نصب نکن. اگر `conda` نصب است:

```bash
conda create -n astrocast39 python=3.9 -y
conda activate astrocast39
python -m pip install 'numpy<2' 'astrocast==0.4.2'
astrocast --help
astrocast detect-events --help
```

در صورت نبود `conda` ابتدا Miniforge را نصب کن یا از محیط Python 3.9 جداگانه استفاده کن. **نصب astroCaST قدیمی روی سیستم‌های جدید ممکن است به‌علت ناسازگاری بعضی کتابخانه‌ها شکست بخورد؛ مراحل نصب در این سیستم هنوز تست نشده‌اند.** نسخه pip برنامه را از `astrocast --help` یا `python -m pip show astrocast` تأیید کن.

## گام ۳: اجرای آشکارساز اصلی روی ۲۸۱ فریمِ کاهش‌یافته

از ریشه پروژه (در محیط astrocast39):

```bash
cd ~/AstroCellLink
bash scripts/run_upstream_astrocast.sh \
  real_data/exvivo_dff_stride2.h5 \
  real_data/exvivo_upstream_v042.roi
```

**نکات:** اسکریپت اگر مسیر خروجی از قبل وجود داشته باشد از بازنویسی جلوگیری می‌کند؛ لاگ، نسخه، checksum و پارامترهای اجرا را ذخیره می‌کند. برخی مقادیر detector پیش‌فرض astroCaST هستند و با آستانه heuristic ما یکسان نیستند. **مقایسه توافق بین الگوریتم‌هاست نه تشخیص درستی.** اگر `astrocast` خطا داد، فایل `real_data/exvivo_upstream_v042.run.log` و متن خطا را ارسال کن. از تغییر تصادفی threshold برای بهتر شدن هم‌پوشانی بدون ثبت پیش‌فرض پرهیز کن.

## گام ۴: مقایسه دقیق برچسب‌های رویدادی (بازگشت به .venv)

```bash
conda deactivate
cd ~/AstroCellLink
source .venv/bin/activate
astrocelllink compare-events \
  --candidate real_data/exvivo_screen/candidate_event_map.tiff \
  --astrocast real_data/exvivo_upstream_v042.roi \
  --min-iou 0.10 --frame 46 \
  --output real_data/detector_comparison
xdg-open real_data/detector_comparison/agreement_preview.png
cat real_data/detector_comparison/comparison.json
```

دستور در صورت عدم یکسان‌بودن ابعاد، **عمداً خطا می‌دهد**؛ فایل‌ها را بدون هم‌ثبت دقیق مکان و زمان Resize نکن. `foreground_jaccard`, `mean_matched_event_iou` و نرخ match، شاخص *توافق الگوریتم‌ها* هستند، نه Precision و Recall روی واقعیت زیستی.

## گام ۵: مسیر شناسایی سلول‌های واقعی (فعلاً بدون Ground Truth)

داده‌های `data/ch0`, `mc/ch0`, `dff/ch0` به‌احتمال زیاد پردازش‌های **همان کانال کلسیمی** هستند؛ کانال مستقل مورفولوژی/رنگ‌آمیزی آستروسیت در این فایل فعلاً تأیید نشده است. در تصاویر شاخه‌دار نمی‌توان با اتکا به روشنایی، مالکیت شاخه‌ها یا هر سلول را اثبات کرد.

اگر تصویر جداگانه **مورفولوژی آستروسیت** داری (مثلاً `structural_astrocytes.tiff`) می‌توان یک Baseline ساخت:

```bash
astrocelllink segment-structural \
  --input structural_astrocytes.tiff \
  --stride 2 --min-area 30 --min-distance 8 \
  --output real_data/segmentation_proposal
xdg-open real_data/segmentation_proposal/proposed_masks_qc.png
```

برای تصویر ساختاری موجود به‌شکل HDF5، از `--dataset YOUR/STRUCTURE/PATH` استفاده کن و `--stride` را مطابق نقشه رویدادی انتخاب کن. **این روش Watershed فقط یک پیشنهاد بصری است** و مدل Deep Learning آموزش‌دیده نیست. اگر فقط کانال کلسیمی داری، از آن برای ادعای شناسایی مستقل سلول‌ها استفاده نکن.

برای Ground Truth، در Fiji/Napari حداقل چند فریم و سلول مشخص را با شناسه‌های جداگانه برچسب بزن (صفر=پس‌زمینه)، یا تصاویر GFAP/ALDH1L1/ترانسژن مستقل با برچسب‌های متخصص تهیه کن. فایل ماسک مرجع را TIFF دوبعدی integer ذخیره کن و با همان مختصات هم‌ثبت کن.

```bash
astrocelllink evaluate-masks \
  --predicted real_data/segmentation_proposal/proposed_instance_masks.tiff \
  --reference /path/to/expert_reference_masks.tiff \
  --iou 0.50 --output real_data/mask_validation
cat real_data/mask_validation/mask_validation.json
```

اگر ماسک‌های واقعاً مناسب و هم‌ثبت فراهم شد، نسبت‌دهی رویداد به هویت سلول با دستور قبلی `astrocelllink run --event-map ... --instance-masks ...` امکان‌پذیر است. نتیجه نیازمند QC زیستی است؛ خطای overlap می‌تواند انتساب اشتباه بسازد.

## پژوهش بعدی

- تکرار توافق detector در چند مجموعه واقعی و چند آستانه ثبت‌شده؛ گزارش رویدادهای split/merge و اختلاف حساسیت
- مجموعه ماسک مرجع با دو متخصص مستقل، توافق بین annotatorها و جداسازی train/test در سطح آزمایش/حیوان
- در صورت وجود ماسک مرجع کافی، مقایسه با Cellpose/StarDist یا آموزش U-Net؛ بدون برچسب، ادعای آموزش مدل جدید نکن
- آزمون مستقل موج کلسیمی پس از کنترل انتخاب رویداد، همبستگی مکانی، کالیبراسیون fps و μm/px
