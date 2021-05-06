import os
import re
import streamlit as st

code = """<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-611FPFGKT5"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'UA-611FPFGKT5');
</script>"""


a = os.path.dirname(st.__file__)+'/static/index.html'
with open(a, 'r') as f:
    data = f.read()
    if len(re.findall('UA-', data)) == 0:
        with open(a, 'w') as ff:
            newdata = re.sub('<head>', '<head>'+code, data)
            ff.write(newdata)
