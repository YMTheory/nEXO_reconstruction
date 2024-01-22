import numpy as np

from nEXO_loader import loader


class diffusion_fitter():
    
    def __init__(self) -> None:
        self.filename = None
        self.loader = loader()
    
    
    def reproduce_waveforms_fromtruth(self, evtid) -> None:
        self.loader.filename = self.filename
        self.loader._load_event()
        
        one_event = self.loader._get_one_event(evtid)
        print(one_event.keys())
        print(f"In event {evtid}, we have total {len(one_event['event_x'])} deposition points and {len(one_event['wf'])} recorded channels.")

        # 1st simple selection w/ charge and noise tag:
        depo_id = np.where(one_event['event_E'] > 0.01 * np.sum(one_event['event_E']))
        cha_id = np.where(one_event['noise_tag'] == 0)
        print(f"After filtering: total {len(depo_id[0])} deposition points and {len(cha_id[0])} non-noisy channel.")


        n_cha_filtered, n_depo_filtered = len(cha_id[0]), len(depo_id[0])
        for i_cha in range(33, 34, 1):
            chaid = cha_id[0][i_cha]
            print(f'The channel is centered at ({one_event["xpos"][chaid]}, {one_event["ypos"][chaid]}). ')
            for i_depo in range(n_depo_filtered):
                depoid = depo_id[0][i_depo]
                cha_type = 'X' if chaid < 16 else "Y"
                print( f"Deposition is at ({one_event['event_x'][depoid]:.3f}, {one_event['event_y'][depoid]:.3f}, {one_event['event_z'][depoid]:.3f}), "
                     f'and channel Id is {chaid} ({cha_type} strip)' \
                    + ', dx = '+f"{np.abs(one_event['xpos'][chaid]-one_event['event_x'][depoid]):.4f}" \
                    + ', dy = '+f"{np.abs(one_event['ypos'][chaid]-one_event['event_y'][depoid]):.4f}" \
                    + ', dE = ' + f"{one_event['event_E'][depoid]:.4f} MeV." )

                if chaid < 16:
                    distance = np.abs(one_event['xpos'][chaid]-one_event['event_x'][depoid])
                else:
                    distance = np.abs(one_event['ypos'][chaid]-one_event['event_y'][depoid])

                
                    