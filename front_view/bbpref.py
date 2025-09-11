# bbperf.py
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median

# ---- 의존 패키지: psutil (필수), pynvml (선택) ----
try:
    import psutil
except Exception:
    print("[ERR] psutil이 필요합니다: pip install psutil", file=sys.stderr)
    sys.exit(1)

# NVIDIA GPU 모니터링 (가능하면 사용)
NV_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    NV_AVAILABLE = True
except Exception:
    NV_AVAILABLE = False

def human_mb(x_bytes):
    return x_bytes / (1024 * 1024)

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def compute_frame_stats(rows):
    """어댑터 JSONL에서 FPS/응답시간(ms)을 계산"""
    if not rows:
        return {"frames": 0, "fps": 0.0, "lat_ms_avg": None, "lat_ms_p95": None}
    # ts는 어댑터에서 time.time()-t0 이므로, 프레임 간 차이를 응답시간으로 사용
    ts = [r["ts"] for r in rows]
    # 연속 프레임 간 간격
    gaps = [ (ts[i]-ts[i-1]) for i in range(1, len(ts)) if ts[i]>=ts[i-1] ]
    fps = (len(rows) / (ts[-1]-ts[0])) if len(rows) >= 2 and (ts[-1]-ts[0])>0 else 0.0
    lat_ms = [g*1000.0 for g in gaps] if gaps else []
    lat_ms_sorted = sorted(lat_ms)
    p95 = None
    if lat_ms_sorted:
        idx = int(round(0.95*(len(lat_ms_sorted)-1)))
        p95 = lat_ms_sorted[idx]
    return {
        "frames": len(rows),
        "fps": fps,
        "lat_ms_avg": mean(lat_ms) if lat_ms else None,
        "lat_ms_p95": p95
    }

def snapshot_gpu(pid):
    """NVIDIA만 지원. 해당 PID가 점유한 GPU 총합(%)와 메모리(MB) 대략 추정"""
    if not NV_AVAILABLE:
        return (0.0, 0.0)
    try:
        dev_count = pynvml.nvmlDeviceGetCount()
        total_util = 0.0
        total_mem = 0.0
        for i in range(dev_count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            total_util += util.gpu  # %
            # 프로세스별 메모리만 뽑으려면 아래 NVML per-process query 필요
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            mem_used_by_pid = 0
            for p in procs:
                if int(getattr(p, 'pid', getattr(p, 'pid_', -1))) == pid:
                    mem_used_by_pid += getattr(p, 'usedGpuMemory', 0)
            if mem_used_by_pid > 0:
                total_mem += mem_used_by_pid / (1024*1024)
        # util은 장치 총합(%)이므로, 다GPU면 평균으로 조정
        if dev_count > 0:
            total_util = total_util / dev_count
        return (float(total_util), float(total_mem))
    except Exception:
        return (0.0, 0.0)

def monitor_process(proc, json_out_path, sample_sec=0.25, mon_out_csv=None, stop_when_json_written=False):
    """
    외부 프로세스 성능 모니터링.
    - CPU% (process), RSS(MB), GPU util%(NVIDIA), GPU mem(MB)
    - 모니터링 결과를 CSV로 저장(옵션)
    - stop_when_json_written=True이면 json_out_path가 생성되면 종료 대기 없이 프로세스만 감시 계속
    """
    p = psutil.Process(proc.pid)
    cpu_hist, rss_hist, gpuu_hist, gpum_hist = [], [], [], []
    t0 = time.time()

    # psutil 첫 CPU% 콜은 기준 설정이므로 버리는 콜 한 번
    try:
        _ = p.cpu_percent(interval=None)
    except Exception:
        pass

    csv_f = None
    if mon_out_csv:
        csv_f = open(mon_out_csv, "w", encoding="utf-8")
        csv_f.write("t,cpu_perc,rss_mb,gpu_util_perc,gpu_mem_mb\n")

    json_written_once = False

    while True:
        if proc.poll() is not None:
            # 프로세스 종료
            break
        try:
            cpu = p.cpu_percent(interval=None)  # non-blocking, 이전 호출 대비 %
            rss = human_mb(p.memory_info().rss)
        except Exception:
            # 프로세스가 이미 종료됐거나 접근 불가
            break
        gpu_util, gpu_mem = snapshot_gpu(proc.pid)

        cpu_hist.append(cpu)
        rss_hist.append(rss)
        gpuu_hist.append(gpu_util)
        gpum_hist.append(gpu_mem)

        t = time.time() - t0
        if csv_f:
            csv_f.write(f"{t:.3f},{cpu:.2f},{rss:.2f},{gpu_util:.2f},{gpu_mem:.2f}\n")

        if (not json_written_once) and json_out_path and Path(json_out_path).exists():
            json_written_once = True
            # 요청 시: JSON이 생성되었다고 모니터링을 멈추지는 않지만 플래그만 둠
            if stop_when_json_written:
                pass

        time.sleep(sample_sec)

    if csv_f:
        csv_f.close()

    def avg(xs): return (sum(xs)/len(xs)) if xs else 0.0
    def p95(xs):
        if not xs: return 0.0
        s = sorted(xs); idx = int(round(0.95*(len(s)-1))); return s[idx]

    return {
        "cpu_avg": avg(cpu_hist),
        "cpu_p95": p95(cpu_hist),
        "rss_avg": avg(rss_hist),
        "rss_peak": max(rss_hist) if rss_hist else 0.0,
        "gpu_util_avg": avg(gpuu_hist),
        "gpu_util_p95": p95(gpuu_hist),
        "gpu_mem_avg": avg(gpum_hist),
        "gpu_mem_peak": max(gpum_hist) if gpum_hist else 0.0,
        "samples": len(cpu_hist)
    }

def run_and_profile(cmdline, json_out, mon_csv=None, timeout=None):
    """
    대상 스크립트를 실행(--json_out으로 JSONL을 내도록 구성되어 있어야 함)하고
    동시 모니터링 후 결과 요약을 반환.
    """
    print(f"[RUN] {cmdline}")
    # Windows 호환을 위해 shell=True 대신 list 사용 권장
    if isinstance(cmdline, str):
        cmd = shlex.split(cmdline, posix=False)
    else:
        cmd = cmdline

    # 기존 json 제거
    if json_out:
        try:
            Path(json_out).unlink(missing_ok=True)
        except Exception:
            pass

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # stdout/stderr는 백그라운드로 흘려보내되, 종료 후 출력
    start = time.time()
    mon = monitor_process(proc, json_out_path=json_out, mon_out_csv=mon_csv)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()

    dur = time.time() - start
    print(f"[END] elapsed: {dur:.2f}s")

    if out:
        print("[STDOUT]\n" + out.strip()[:2000])
    if err:
        print("[STDERR]\n" + err.strip()[:2000])

    rows = []
    if json_out and Path(json_out).exists():
        rows = load_jsonl(json_out)
    frame_stats = compute_frame_stats(rows)
    return mon, frame_stats, dur

def summarize(tag, mon, frame_stats, dur):
    print(f"\n=== {tag} ===")
    print(f"- Elapsed (s)           : {dur:.2f}")
    print(f"- Frames                : {frame_stats['frames']}")
    print(f"- FPS (from JSONL)      : {frame_stats['fps']:.2f}")
    if frame_stats['lat_ms_avg'] is not None:
        print(f"- Latency avg (ms/frame): {frame_stats['lat_ms_avg']:.2f}")
        print(f"- Latency p95 (ms/frame): {frame_stats['lat_ms_p95']:.2f}")
    print(f"- CPU avg / p95 (%)     : {mon['cpu_avg']:.2f} / {mon['cpu_p95']:.2f}")
    print(f"- RSS avg / peak (MB)   : {mon['rss_avg']:.2f} / {mon['rss_peak']:.2f}")
    print(f"- GPU util avg/p95 (%)  : {mon['gpu_util_avg']:.2f} / {mon['gpu_util_p95']:.2f}")
    print(f"- GPU mem avg/peak (MB) : {mon['gpu_mem_avg']:.2f} / {mon['gpu_mem_peak']:.2f}")
    print(f"- Monitor samples       : {mon['samples']}")

def compare_reports(repA, repB, tagA="PREV", tagB="LITE"):
    # 간단 비교(차이 출력). rep = (mon, frame_stats, dur)
    monA, fsA, durA = repA
    monB, fsB, durB = repB

    def diff(a,b): 
        if a is None or b is None: return None
        return b-a

    print("\n=== COMPARE ===")
    print(f"FPS: {fsA['fps']:.2f} -> {fsB['fps']:.2f}  (Δ {diff(fsA['fps'], fsB['fps']):+.2f})")
    if fsA['lat_ms_avg'] is not None and fsB['lat_ms_avg'] is not None:
        print(f"Latency avg(ms): {fsA['lat_ms_avg']:.2f} -> {fsB['lat_ms_avg']:.2f}  (Δ {diff(fsA['lat_ms_avg'], fsB['lat_ms_avg']):+.2f})")
        print(f"Latency p95(ms): {fsA['lat_ms_p95']:.2f} -> {fsB['lat_ms_p95']:.2f}  (Δ {diff(fsA['lat_ms_p95'], fsB['lat_ms_p95']):+.2f})")
    print(f"CPU avg(%): {monA['cpu_avg']:.2f} -> {monB['cpu_avg']:.2f}  (Δ {diff(monA['cpu_avg'], monB['cpu_avg']):+.2f})")
    print(f"RSS peak(MB): {monA['rss_peak']:.2f} -> {monB['rss_peak']:.2f}  (Δ {diff(monA['rss_peak'], monB['rss_peak']):+.2f})")
    print(f"GPU util avg(%): {monA['gpu_util_avg']:.2f} -> {monB['gpu_util_avg']:.2f}  (Δ {diff(monA['gpu_util_avg'], monB['gpu_util_avg']):+.2f})")
    print(f"GPU mem peak(MB): {monA['gpu_mem_peak']:.2f} -> {monB['gpu_mem_peak']:.2f}  (Δ {diff(monA['gpu_mem_peak'], monB['gpu_mem_peak']):+.2f})")

def main():
    ap = argparse.ArgumentParser(description="Black-box Performance Runner/Reporter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # run+report 한 번에
    runp = sub.add_parser("run", help="하나의 커맨드 실행 & 성능 리포트")
    runp.add_argument("--cmdline", required=True, help="실행 커맨드(따옴표로 감싸기)")
    runp.add_argument("--json_out", required=True, help="어댑터가 생성할 JSONL 파일 경로")
    runp.add_argument("--mon_csv", default=None, help="모니터 CSV 저장 경로(선택)")
    runp.add_argument("--timeout", type=float, default=None)

    # 두 개를 연속 실행하고 비교
    cmp = sub.add_parser("compare", help="두 커맨드 연속 실행 & 비교 요약")
    cmp.add_argument("--cmd_prev", required=True)
    cmp.add_argument("--json_prev", required=True)
    cmp.add_argument("--cmd_lite", required=True)
    cmp.add_argument("--json_lite", required=True)
    cmp.add_argument("--mon_prev_csv", default=None)
    cmp.add_argument("--mon_lite_csv", default=None)
    cmp.add_argument("--timeout", type=float, default=None)

    # 이미 실행된 결과 파일들만으로 리포트
    rep = sub.add_parser("report", help="이미 생성된 결과(JSONL/CSV)만으로 리포트")
    rep.add_argument("--json", required=True)
    rep.add_argument("--mon_csv", default=None)

    args = ap.parse_args()

    if args.cmd == "run":
        mon, fs, dur = run_and_profile(args.cmdline, args.json_out, args.mon_csv, args.timeout)
        summarize("RUN", mon, fs, dur)

    elif args.cmd == "compare":
        repA = run_and_profile(args.cmd_prev, args.json_prev, args.mon_prev_csv, args.timeout)
        summarize("PREV", *repA)
        repB = run_and_profile(args.cmd_lite, args.json_lite, args.mon_lite_csv, args.timeout)
        summarize("LITE", *repB)
        compare_reports(repA, repB)

    elif args.cmd == "report":
        # 모니터 CSV 없이도 psutil 수치는 알 수 없으니, CSV 없으면 프레임/지연만 출력
        rows = load_jsonl(args.json)
        fs = compute_frame_stats(rows)
        print("=== REPORT ===")
        print(f"- Frames           : {fs['frames']}")
        print(f"- FPS              : {fs['fps']:.2f}")
        if fs['lat_ms_avg'] is not None:
            print(f"- Latency avg (ms) : {fs['lat_ms_avg']:.2f}")
            print(f"- Latency p95 (ms) : {fs['lat_ms_p95']:.2f}")
        if args.mon_csv and Path(args.mon_csv).exists():
            # CSV를 읽어 간단 평균/피크만 출력(선택)
            import csv
            cpu, rss, gu, gm = [], [], [], []
            with open(args.mon_csv, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    cpu.append(float(r["cpu_perc"]))
                    rss.append(float(r["rss_mb"]))
                    gu.append(float(r["gpu_util_perc"]))
                    gm.append(float(r["gpu_mem_mb"]))
            def avg(xs): return (sum(xs)/len(xs)) if xs else 0.0
            print(f"- CPU avg (%)      : {avg(cpu):.2f}")
            print(f"- RSS peak (MB)    : {max(rss) if rss else 0.0:.2f}")
            print(f"- GPU util avg (%) : {avg(gu):.2f}")
            print(f"- GPU mem peak (MB): {max(gm) if gm else 0.0:.2f}")

if __name__ == "__main__":
    main()
