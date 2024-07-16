import typing
import importlib


if typing.TYPE_CHECKING:
    from . import (
        # train_byol_single,
        # train_dtc_local,
        # train_dtc_single_chapman,
        # train_dtc_single,
        # train_dtc_vae_single,
        # train_dtc,
        # train_dtcr_local,
        # train_dtcr_single,
        train_fedbyol,
        train_fedu,
        train_fedema,
        train_fedsimsiam,
        train_fedsimclr,
        # train_fedbyolG,
        train_fedx,
        # train_fedGmod,
        train_fedorche,
        # train_fedorcheG,
        # train_fedorcheGnoT,
        train_fedbyolDink,
        train_fedbyolSRI,
        # train_fedorcheGnoTnoProj,
        # train_fedorcheGHGN,
        # train_fedorcheGProxy,
        # train_fedswav,
        # train_fedsela,
        # train_gmm_vae_single,
        # train_dtcr,
        train_LDAWA,
        train_fedU2
    )
else:
    def __getattr__(name: str):
        return importlib.import_module(f'.{name}', __name__)
