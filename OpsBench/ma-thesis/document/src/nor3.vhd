library IEEE;
use IEEE.std_logic_1164.all;

entity nor3 is
	port (	a, b, c : in 	std_logic;
			y : out 	std_logic);
end entity nor3;

architecture BEHAVIORAL of nor3 is
signal m: std_logic;
begin
	m <= a or b or c;
	y <= not m;
end architecture BEHAVIORAL;
