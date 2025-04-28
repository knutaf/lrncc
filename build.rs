use std::path::Path;

fn add_tests_from_dir(
    dir: &Path,
    indent: &str,
    testfile: &mut impl std::io::Write,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let filename = path.file_name().unwrap().to_str().unwrap();

        // Each subdirectory goes into its own module.
        if path.is_dir() {
            writeln!(
                testfile,
                r"{}mod {} {{
{}    use super::*;
",
                indent, filename, indent
            )
            .unwrap();

            add_tests_from_dir(&path, &format!("{}    ", indent), testfile)?;

            writeln!(testfile, "{}}}\n", indent).unwrap();
        } else {
            let stem = path.file_stem().unwrap().to_str().unwrap();

            // Generate the code to run a test from this file path, named after the base name of the .c file.
            writeln!(
                testfile,
                r###"{}test!(r#{} {{ run_test_from_file(r"{}"); }});"###,
                indent,
                stem,
                path.display()
            )
            .unwrap()
        }
    }

    Ok(())
}

fn main() -> std::io::Result<()> {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let destination = std::path::Path::new(&out_dir).join("test.rs");
    let mut testsfile = std::fs::File::create(&destination).unwrap();

    let testfiles_root = std::path::Path::new("testfiles");
    add_tests_from_dir(&testfiles_root, "", &mut testsfile)?;
    Ok(())
}
