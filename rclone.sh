module load rclone/1.63.1
rclone -v copy ./ pawsey1001_rakib:pc-srgan/ \
--include "OTHERS/**" \
--include "data/**" \
--include "tmp/**" \
--include "bash_logs/**" \
--include "results/**" \
