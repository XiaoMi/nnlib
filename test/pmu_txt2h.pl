#!/usr/bin/perl
use strict;


print "const char *event_names[] = {\n";

my @PMU_names = ('CYCLES');
my @PMU_descrs = ('Cycle count');
my $MaxPMU = -1;
while (my $line = <>) {
  if ($line=~m/(\d+)\s+(\S+)\s+(.*)/) {
	my $idx = $1;
	my $name = $2;
	my $descr = $3;
	if ($idx > $MaxPMU) {
	  $MaxPMU = $idx;
	}
	$PMU_names[$idx] = $name;
	$PMU_descrs[$idx] = $descr;
  }
}

for (my $i=0; $i<$MaxPMU; $i++) {
  printf("/*%04d 0x%04x*/ \"%s\",\n", $i, $i, $PMU_names[$i]);
}

print "};\n";
