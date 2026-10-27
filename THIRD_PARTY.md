# Third-party materials and optional presentation tools

The repository release omits locally downloaded LI-COR manuals, vendor webpages,
the downloaded Saathoff and Welles paper, commercial Areal font files, poster
logos/branding, and temporary renders. Their presence in a private working
snapshot does not confer redistribution rights. Consult [source links](docs/REFERENCES.md).

Historical poster and PDF builders are retained as source, but are **not part of
the supported reproduction pipeline**. They reference a separate blog-content
checkout and local fonts/assets. To use them, supply properly licensed assets
and adapt the paths. Their optional dependencies include reportlab, pypdf, qrcode,
fontTools, Pillow, and skia-python; these are not required for the scientific
workflow. The artistic poster does not have to be reproduced to reproduce the data.

Scientific output plots use their original rendering conventions. Third-party
libraries retain their own licenses. References to LI-COR, Astera, GW, NSF, and
other organizations do not imply endorsement.
