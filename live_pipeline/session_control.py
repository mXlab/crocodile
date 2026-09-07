"""Send one session-control OSC message to live_pipeline.py and exit.

Stands in for a real operator control surface (a physical panel, a
TouchOSC/Processing app) until one exists -- mirrors
replay_biodata_as_osc.py's role of standing in for real sensor hardware.
One action per invocation, for repeatable manual test sequences and,
until a real control surface exists, as the actual way to operate a
session by hand.

Usage:
    python live_pipeline/session_control.py --start-session [ID]
    python live_pipeline/session_control.py --start-calibration
    python live_pipeline/session_control.py --stop-calibration
    python live_pipeline/session_control.py --start-live
    python live_pipeline/session_control.py --recalibrate
    python live_pipeline/session_control.py --end-session
"""

import argparse

from pythonosc.udp_client import SimpleUDPClient


def main():
    parser = argparse.ArgumentParser(
        description='Send one session-control OSC message to live_pipeline.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9000, help="live_pipeline.py's --in-port")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--start-session', nargs='?', const='', metavar='ID',
                        help='Start a new session. Optional session ID; auto-generated if omitted.')
    action.add_argument('--start-calibration', action='store_true')
    action.add_argument('--stop-calibration', action='store_true')
    action.add_argument('--start-live', action='store_true')
    action.add_argument('--recalibrate', action='store_true',
                        help='Refresh the SCR threshold in place, without interrupting LIVE output')
    action.add_argument('--end-session', action='store_true')
    args = parser.parse_args()

    client = SimpleUDPClient(args.host, args.port)

    if args.start_session is not None:
        address, osc_args = '/crocodile/session/start', ([args.start_session] if args.start_session else [])
    elif args.start_calibration:
        address, osc_args = '/crocodile/calibration/start', []
    elif args.stop_calibration:
        address, osc_args = '/crocodile/calibration/stop', []
    elif args.start_live:
        address, osc_args = '/crocodile/live/start', []
    elif args.recalibrate:
        address, osc_args = '/crocodile/calibration/recalibrate', []
    elif args.end_session:
        address, osc_args = '/crocodile/session/end', []

    client.send_message(address, osc_args)
    print(f"Sent {address} {osc_args} to {args.host}:{args.port}")


if __name__ == '__main__':
    main()
