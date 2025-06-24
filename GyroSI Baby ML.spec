# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['/Users/basil/Development/BabyLM/src/main.py'],
    pathex=[],
    binaries=[],
    datas=[('/Users/basil/Development/BabyLM/src/frontend/assets', 'assets')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='GyroSI Baby ML',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['/Users/basil/Development/BabyLM/src/frontend/assets/icons/mingcute--baby-fill.png'],
)
app = BUNDLE(
    exe,
    name='GyroSI Baby ML.app',
    icon='/Users/basil/Development/BabyLM/src/frontend/assets/icons/mingcute--baby-fill.png',
    bundle_identifier='com.gyrosi.babylm',
)
