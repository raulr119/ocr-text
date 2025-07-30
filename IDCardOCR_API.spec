# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('app', 'app'), ('models', 'models'), ('.env', '.')]
binaries = [('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\common.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\libblas.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\libgcc_s_seh-1.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\libgfortran-3.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\libiomp5md.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\liblapack.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\libquadmath-0.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\mkldnn.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\mklml.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\warpctc.dll', '.'), ('D:\\envs\\py310\\lib\\site-packages\\paddle\\libs\\warprnnt.dll', '.')]
hiddenimports = ['fastapi.routing', 'uvicorn.config', 'uvicorn.supervisors.basereload', 'uvicorn.supervisors.watchgod', 'uvicorn.supervisors.statreload', 'torch._C', 'torch._C._jit_tree_views', 'torch.nn.functional', 'ultralytics.nn.modules', 'ultralytics.nn.tasks', 'paddleocr.ppocr.utils.logging', 'paddleocr.tools.infer']
tmp_ret = collect_all('paddleocr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pydantic_settings')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('python_dotenv')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('fastapi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('uvicorn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('Cython')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyclipper')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('timm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app\\main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='IDCardOCR_API',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
