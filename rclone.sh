module load rclone/1.63.1
rclone -v copy ./ pawsey1001_rakib:pc-srgan/ \
--include "OTHERS/**" \
--include "data/Allen-Cahn_Periodic.zip" \
--include "tmp/**" \
--include "bash_logs/**" \
--include "results/all_test_results.csv" \
# --include "results/**" \
