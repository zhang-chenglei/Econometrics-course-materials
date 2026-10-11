* 第六讲依赖安装：已有命令直接核验，不覆盖已有版本。
set more off
foreach cmd in ftools reghdfe mmerge outreg outreg2 {
    capture which `cmd'
    if _rc ssc install `cmd'
    which `cmd'
}
capture which cgmreg
if _rc {
    local plus : sysdir PLUS
    capture mkdir "`plus'c"
    copy "https://cameron.econ.ucdavis.edu/research/cgmreg.ado" "`plus'c/cgmreg.ado"
}
which cgmreg
which frmttable
display "L6_DEPENDENCIES_OK"
