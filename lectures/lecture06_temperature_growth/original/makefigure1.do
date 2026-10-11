clear
clear matrix
set more off
set matsize 800
set mem 200m
capture log close

use climate_panel, clear
keep if year == 2000
g lngdp2000 = ln(rgdpl)

keep fips60_06 parent lngdp2000 
tempfile tempgdp
save `tempgdp'	

use climate_panel, clear
g lngdpwdi = ln(gdpLCU)

* collapse to parent level weighting by 2000pop
mmerge fips60_06 using `tempgdp',type(n:1)

*calculate GDP growth (WDI)
encode fips60_06, g(cc_num) 
sort country_code year 
tsset cc_num year 
gen temp1 = l.lngdpwdi
gen g=lngdpwdi-temp1
replace g = g * 100 
drop temp1
summarize g

* Drop if less than 20 yrs of GDP data
g tempnonmis = 1 if g != .
replace tempnonmis = 0 if g == .
bys fips60_06: egen tempsumnonmis = sum(tempnonmis)
drop if tempsumnonmis  < 20

sort parent

foreach Xvar of var wtem wpre {
	by parent: egen `Xvar'max = max(`Xvar')
	by parent: egen `Xvar'min = min(`Xvar')
	by parent: egen `Xvar'temp50s = mean(`Xvar') if year >= 1950 & year <=1959
	by parent: egen `Xvar'temp00s= mean(`Xvar') if year >= 1996 & year <=2005
	by parent: egen `Xvar'50s = mean(`Xvar'temp50s)
	by parent: egen `Xvar'00s = mean(`Xvar'temp00s)
				
	label var `Xvar'50s "1950-1959"
	label var `Xvar'00s "1996-2005"
}

label var lngdp2000 "Log per-capita GDP in 2000"
bys parent: keep if _n == 1
save temp,replace
twoway (rspike wtemmax wtemmin lngdp2000, lcolor(gs12)) (scatter wtem50s lngdp2000, mcolor(blue) msymbol(circle_hollow) msize(.75) mlabsize(tiny) mlabcolor(black) mlabel(country_code)) (scatter wtem00s lngdp2000, msize(.75) mcolor(red) msymbol(plus)) , legend(order(2 3) label(2 "Mean 1950-1959") label( 3 "Mean 1996-2005")) graphregion(color(white)) ylab(-10(10)30, nogrid)subtitle("Weighted by Population", position(11) size(small) color(black))  title("Temperature",position(11)) saving(rspikewtem.gph,replace) ytitle("degrees")
graph export `filename'-temp-presentation.eps,replace

twoway (rspike wpremax wpremin lngdp2000, lcolor(gs12)) (scatter wpre50s lngdp2000, mcolor(blue) msymbol(circle_hollow) msize(.75) mlabsize(tiny) mlabcolor(black) mlabel(country_code)) (scatter wpre00s lngdp2000, msize(.75) mcolor(red) msymbol(plus)) , legend(order(2 3) label(2 "Mean 1950-1959") label( 3 "Mean 1996-2005")) graphregion(color(white)) ylab(0(20)60, nogrid)subtitle("Weighted by Population", position(11) size(small) color(black))  title("Precipitation",position(11)) saving(rspikewpre.gph,replace) ytitle("100s mm / year")
graph export `filename'-precip-presentation.eps,replace
