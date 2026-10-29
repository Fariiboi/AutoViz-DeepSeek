@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "ROOT=%~dp0"
set "PG_BIN=%ROOT%workspace\pgsql\pgsql\bin"
set "PGDATA=%ROOT%workspace\pgdata"
set "LOG=%ROOT%workspace\pgvector.log"
set "PORT=5433"

if not exist "%PG_BIN%\pg_ctl.exe" (
    echo [错误] 未找到便携版 PostgreSQL：%PG_BIN%
    goto :end
)
if not exist "%PGDATA%\PG_VERSION" (
    echo [错误] 未找到数据目录：%PGDATA%
    goto :end
)

call :port_in_use
if not errorlevel 1 (
    echo 数据库已在运行中（端口 %PORT% 已被占用）。
    goto :end
)

echo 正在启动便携版 PostgreSQL（端口 %PORT%）...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\start_pg.ps1" -Root "%~dp0." >nul 2>&1

set /a tries=0
:wait_loop
call :port_in_use
if not errorlevel 1 (
    echo 启动成功：PostgreSQL 正在 127.0.0.1:%PORT% 运行。
    goto :end
)
set /a tries+=1
if %tries% GEQ 20 (
    echo [错误] 启动超时，请检查日志：%LOG%
    goto :end
)
ping -n 2 127.0.0.1 >nul 2>&1
goto :wait_loop

:port_in_use
netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul 2>&1
exit /b %errorlevel%

:end
if /I not "%AUTOVIZ_NO_PAUSE%"=="1" pause
endlocal
