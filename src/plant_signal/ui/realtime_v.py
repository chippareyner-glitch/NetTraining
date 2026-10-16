"""Streamlit entry for the V-prefix serial protocol variant."""
from plant_signal.ui.realtime import main


if __name__ == '__main__':
    main(enable_v_format=True)
