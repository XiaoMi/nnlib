#!/usr/bin/perl
use strict;


# Generate hexagon_nn.idl from hexagon_nn.proto_idl
#
# Why the indirection?
#    idl does not support #include,
#    and we need to support "domains" and "non-domains"
#    which would otherwise require copy-pasta.
#    So... To reduce bugs, this script boils the copy-pasta.

my $INFILE  = "interface/hexagon_nn.proto_idl";
my $OUTFILE = "interface/hexagon_nn.idl";



open IN,$INFILE or die "ERROR: Could not open $INFILE for reading\n";
open OUT,">$OUTFILE" or die "ERROR: Could not open $OUTFILE for writing\n";

print OUT <<EOF;
/*
    ______        _   _       _     _____    _ _ _
    |  _  \      | \ | |     | |   |  ___|  | (_) |
    | | | |___   |  \| | ___ | |_  | |__  __| |_| |_
    | | | / _ \  | . ` |/ _ \| __| |  __|/ _` | | __|
    | |/ / (_) | | |\  | (_) | |_  | |__| (_| | | |_
    |___/ \___/  \_| \_/\___/ \__| \____/\__,_|_|\__|

    This is a generated file, from $0 operating on $INFILE
*/
EOF


my $in_block = 0;
my $block = '';
while (my $line = <IN>) {

    print OUT $line;

    if ($in_block && $line=~m/\}/) {
        $in_block = 0;
		print OUT '#include "remote.idl"'."\n\n";
        print OUT "interface hexagon_nn_domains : remote_handle64 {\n";
        print OUT $block;
        print OUT "}; /* interface hexagon_nn_domains */\n";
    }
	if ($in_block) {
		$block.=$line;
	}
    if ($line=~m/interface hexagon_nn/) {
        $in_block = 1;
    }
}
close OUT;
