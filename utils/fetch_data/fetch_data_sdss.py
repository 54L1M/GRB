import sys
import numpy as np
from astroML.datasets.tools import sql_query
#from utils.npy2csv import npy2csv

NOBJECTS = 1000000

GAL_COLORS_NAMES = ["u", "g", "r", "i", "z", "specClass", "redshift", "redshift_err"]

ARCHIVE_FILE = "sdss_galaxy_colors.npy"


query_text = "\n".join(
    (
        "SELECT TOP %i" % NOBJECTS,
        "  p.u, p.g, p.r, p.i, p.z, s.class, s.z, s.zerr",
        "FROM PhotoObj AS p",
        "  JOIN SpecObj AS s ON s.bestobjid = p.objid",
        "WHERE ",
        "  p.u BETWEEN 0 AND 19.6",
        "  AND p.g BETWEEN 0 AND 20",
        "  AND s.class <> 'UNKNOWN'",
        "  AND s.class <> 'STAR'",
        "  AND s.class <> 'SKY'",
        "  AND s.class <> 'STAR_LATE'",
    )
)

output = sql_query(query_text)

kwargs = {
    "delimiter": ",",
    "skip_header": 2,
    "names": GAL_COLORS_NAMES,
    "dtype": None,
}

if sys.version_info[0] >= 3:
    kwargs["encoding"] = "ascii"

data = np.genfromtxt(output, **kwargs)
np.save(f"data\sdss\sdss_galaxy_{NOBJECTS}.npy", data)

#npy2csv(f"data\sdss\sdss_galaxy_{NOBJECTS}.npy")
# data = fetch_sdss_galaxy_colors(download_if_missing=False)
# np.save("sdss_galaxy.npy", data)
