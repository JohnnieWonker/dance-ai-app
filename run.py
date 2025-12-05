import os
import sys
import subprocess
import time
import webbrowser


def resource_path(relative_path: str) -> str:
    """
    兼容 PyInstaller 的资源路径：
    - 开发环境：返回当前文件所在目录 + relative_path
    - 打包后：返回 _MEIPASS 临时目录 + relative_path
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def main():
    port = "8501"

    # 找到打包后的 pro_app.py 真实路径
    app_path = resource_path(os.path.join("dance_ai_app", "pro_app.py"))

    # 以当前 EXE 自带的 Python 解释器运行 streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.headless=true",
        f"--server.port={port}",
    ]

    # 工作目录设为 pro_app.py 所在目录，避免相对路径问题
    workdir = os.path.dirname(app_path)

    # 启动子进程，不阻塞主程序
    subprocess.Popen(cmd, cwd=workdir)

    # 稍微等一下，让服务起来
    time.sleep(2)

    # 打开浏览器
    webbrowser.open(f"http://localhost:{port}")


if __name__ == "__main__":
    main()
