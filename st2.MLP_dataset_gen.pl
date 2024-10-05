#!/usr/bin/perl -w

use strict;
use warnings;
use List::Util qw(shuffle);
use Getopt::Long;
use File::Path qw(make_path);

# Default values
my $input_nodes = './graph_model/nodes.csv';
my $input_edges = './graph_model/edges.csv';
my $input_embeddings = './graph_model/node_embeddings.128_64.csv';
my $output_ppi_learning_set = './MLP_classifier/dataset/';

# Get command-line arguments
GetOptions(
    'input_nodes=s'             => \$input_nodes,
    'input_edges=s'             => \$input_edges,
    'input_embeddings=s'        => \$input_embeddings,
    'output_ppi_learning_set=s' => \$output_ppi_learning_set,
);

my %nodes_dict;
my %edges_ppi;

# Read nodes file
open(my $nodes_fh, '<', $input_nodes) or die "Cannot open $input_nodes: $!";
while (<$nodes_fh>) {
    chomp;
    if (/^(\d+),.*?,(2A\d+)$/) {
        $nodes_dict{$1} = $2;
    }
}
close($nodes_fh);

# Read embeddings file
open(my $embeddings_fh, '<', $input_embeddings) or die "Cannot open $input_embeddings: $!";
while (<$embeddings_fh>) {
    chomp;
    if (/^(\d+),(.*?)$/) {
        my $node_id = $1;
        my $node_vec = $2;
        if (exists $nodes_dict{$node_id}) {
            $nodes_dict{$node_id} = $node_vec;
        }
    }
}
close($embeddings_fh);

# Read edges file
open(my $edges_fh, '<', $input_edges) or die "Cannot open $input_edges: $!";
while (<$edges_fh>) {
    chomp;
    if (/^\d+,(\d+),(\d+),.*?,(.*?)$/) {
        my $l_obj_id = $1;
        my $r_obj_id = $2;
        my $coocc_val = $3;

        if (exists $nodes_dict{$l_obj_id} && exists $nodes_dict{$r_obj_id} && !exists $edges_ppi{"$l_obj_id\_$r_obj_id"}) {
            $edges_ppi{"$l_obj_id\_$r_obj_id"} = "$nodes_dict{$l_obj_id},$nodes_dict{$r_obj_id},$coocc_val";
            $edges_ppi{"$r_obj_id\_$l_obj_id"} = "$nodes_dict{$r_obj_id},$nodes_dict{$l_obj_id},$coocc_val";
        }
    }
}
close($edges_fh);

# Generate random pairs for negative examples
sub generate_random_pairs {
    my ($num_pairs, $nodes_ref, $edges_ref) = @_;
    my @nodes = keys %$nodes_ref;
    my @random_pairs;
    my %seen_pairs;

    while (scalar(@random_pairs) < $num_pairs) {
        my $node1 = $nodes[int(rand(@nodes))];
        my $node2 = $nodes[int(rand(@nodes))];
        next if $node1 == $node2 || exists $edges_ref->{"$node1\_$node2"} || exists $seen_pairs{"$node1\_$node2"};

        my $rand = rand();
        my $coocc_val = ($rand < 0.3) ? rand() : 0;

        push @random_pairs, "$node1\_$node2,$nodes_ref->{$node1},$nodes_ref->{$node2},$coocc_val";
        $seen_pairs{"$node1\_$node2"} = 1;
        $seen_pairs{"$node2\_$node1"} = 1;
    }
    return @random_pairs;
}

# Create datasets
sub create_dataset {
    my ($num_pairs, $edges_ref, $nodes_ref) = @_;
    my @positive_pairs = (shuffle keys %$edges_ref)[0 .. $num_pairs - 1];
    my @negative_pairs = generate_random_pairs($num_pairs, $nodes_ref, $edges_ref);
    return (\@positive_pairs, \@negative_pairs);
}

my ($train_positives, $train_negatives) = create_dataset(200_000, \%edges_ppi, \%nodes_dict);
my ($val_positives, $val_negatives)     = create_dataset(20_000, \%edges_ppi, \%nodes_dict);
my ($test_positives, $test_negatives)   = create_dataset(10_000, \%edges_ppi, \%nodes_dict);

# Ensure output directory exists
unless (-d $output_ppi_learning_set) {
    make_path($output_ppi_learning_set) or die "Failed to create path: $output_ppi_learning_set";
}

# Print to output files
sub print_dataset {
    my ($filename, $positives_ref, $negatives_ref) = @_;
    open(my $fh, '>', $filename) or die "Cannot open $filename: $!";
    foreach my $pair_id (@$positives_ref) {
        print $fh "$pair_id,$edges_ppi{$pair_id},1\n";  # 1 for positive
    }
    foreach my $pair_line (@$negatives_ref) {
        print $fh "$pair_line,0\n";  # 0 for negative
    }
    close($fh);
}

print_dataset("$output_ppi_learning_set/st2.ppi_training_set.csv", $train_positives, $train_negatives);
print_dataset("$output_ppi_learning_set/st2.ppi_validation_set.csv", $val_positives, $val_negatives);
print_dataset("$output_ppi_learning_set/st2.ppi_testing_set.csv", $test_positives, $test_negatives);
