# آپلود AstroCellLink در GitHub با ترمینال (Linux Mint)

فایل ZIP را در `~/Downloads/` ذخیره کن و دستورهای زیر را از ترمینال اجرا کن. **در پروژه‌ی اصلی `~/AstroCellLink` دستور `git add .` اجرا نکن**؛ این روش کد منتشرشدنی را در یک مسیر جدا آپلود می‌کند تا فایل‌های بزرگ `real_data` اشتباهی ارسال نشوند.

```bash
set -e
ZIP="$HOME/Downloads/AstroCellLink_GitHub_Publish_v0.3.0.zip"
WORK=$(mktemp -d "$HOME/Downloads/astrocelllink_publish_XXXXXX")
git clone https://github.com/sanatat/AstroCellLink.git "$WORK/repo"
unzip -q "$ZIP" -d "$WORK/repo"
cd "$WORK/repo"
test -f README.md && test -d astrocelllink && test -f pyproject.toml
git add -A
git status --short
git commit -m "Publish AstroCellLink v0.3.0 research prototype"
git push -u origin HEAD:main
printf '\nRepository: https://github.com/sanatat/AstroCellLink\n'
```

اگر `git commit` با خطای *Please tell me who you are* مواجه شد، در پوشه‌ی `$WORK/repo` نام و ایمیل دلخواه خودت را تنظیم کن و سپس commit/push را تکرار کن:

```bash
git config user.name "YOUR_DISPLAY_NAME"
git config user.email "YOUR_EMAIL"
git commit -m "Publish AstroCellLink v0.3.0 research prototype"
git push -u origin HEAD:main
```

برای احراز هویت در `git push`، GitHub رمز عادی حساب را نمی‌پذیرد. اگر ابزار `gh` را داری می‌توانی `gh auth login` را اجرا کنی، در غیر این صورت از Personal Access Token یا Credential Manager استفاده کن. اگر مخزن روی GitHub قبلاً commit دارد، بعد از `git clone` فایل‌های مشابه با نسخه‌ی ZIP جایگزین می‌شوند؛ بنابراین `git status --short` را قبل از commit بررسی کن.

اطمینان: README.md و LICENSE و pyproject.toml باید در **ریشه** مخزن قرار بگیرند.
