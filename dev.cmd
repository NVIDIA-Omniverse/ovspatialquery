@rem SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
@rem SPDX-License-Identifier: Apache-2.0

@echo off
setlocal
set "OVSPATIALQUERY_ENTRYPOINT=dev.cmd"
set "OVSPATIALQUERY_ROOT=%~dp0"
set "_OVSPATIALQUERY_FORWARDED_ARGS="

:collect_arguments
if "%~1"=="" goto launch
set _OVSPATIALQUERY_FORWARDED_ARGS=%_OVSPATIALQUERY_FORWARDED_ARGS% "%~1"
shift
goto collect_arguments

:launch
call "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%OVSPATIALQUERY_ROOT%dev.ps1" %_OVSPATIALQUERY_FORWARDED_ARGS%
set "_OVSPATIALQUERY_EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %_OVSPATIALQUERY_EXIT_CODE%
