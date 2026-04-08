use super::test_harness::temp_dir;
use super::*;

#[test]
fn kernelspec_generation_captures_binary_env_and_interrupt_mode() {
    let options = KernelSpecOptions {
        kernel_name: "axioma-test".to_string(),
        display_name: "Axioma Test".to_string(),
        binary_path: Some(PathBuf::from("/tmp/axioma-jupyter")),
        working_dir: Some(PathBuf::from("/work/notebooks")),
        std_path: Some(PathBuf::from("/work/std")),
        prefix: None,
        user: false,
    };

    let spec = build_kernelspec(&options).expect("kernelspec");
    assert_eq!(spec.argv, vec!["/tmp/axioma-jupyter", "{connection_file}"]);
    assert_eq!(spec.display_name, "Axioma Test");
    assert_eq!(spec.language, "axioma");
    assert_eq!(spec.interrupt_mode.as_deref(), Some("message"));
    assert_eq!(
        spec.env.get("AXIOMA_JUPYTER_WORKDIR").and_then(Value::as_str),
        Some("/work/notebooks")
    );
    assert_eq!(
        spec.env.get("AXIOMA_STD_PATH").and_then(Value::as_str),
        Some("/work/std")
    );
}

#[test]
fn kernelspec_install_dir_honors_user_and_prefix_modes() {
    let mut options = default_kernelspec_options();
    options.user = true;
    let home = PathBuf::from("/tmp/axioma-home");
    let user_dir = kernelspec_install_dir(&options, Some(home.clone())).expect("user dir");
    assert!(user_dir.ends_with(PathBuf::from("axioma")));
    assert!(user_dir.starts_with(user_kernels_dir_from_home(&home)));

    options.user = false;
    options.prefix = Some(PathBuf::from("/opt/axioma"));
    let prefix_dir = kernelspec_install_dir(&options, None).expect("prefix dir");
    assert_eq!(
        prefix_dir,
        PathBuf::from("/opt/axioma/share/jupyter/kernels/axioma")
    );
}

#[test]
fn install_kernelspec_writes_valid_kernel_json() {
    let prefix = temp_dir("kernelspec-install");
    let options = KernelSpecOptions {
        kernel_name: "axioma".to_string(),
        display_name: "Axioma".to_string(),
        binary_path: Some(PathBuf::from("/tmp/axioma-jupyter")),
        working_dir: Some(PathBuf::from("/tmp/notebooks")),
        std_path: Some(PathBuf::from("/tmp/std")),
        prefix: Some(prefix.clone()),
        user: false,
    };

    let install_dir = install_kernelspec(&options).expect("install kernelspec");
    let kernel_json_path = install_dir.join("kernel.json");
    let written = fs::read_to_string(&kernel_json_path).expect("read kernel.json");
    let parsed: KernelSpec = serde_json::from_str(&written).expect("parse kernel.json");
    assert_eq!(parsed.display_name, "Axioma");
    assert_eq!(parsed.argv[0], "/tmp/axioma-jupyter");
    assert_eq!(parsed.interrupt_mode.as_deref(), Some("message"));
}

#[test]
fn cli_parsing_supports_install_and_connection_file_modes() {
    let install = parse_cli_command(&[
        "axioma-jupyter".to_string(),
        "install".to_string(),
        "--user".to_string(),
        "--name".to_string(),
        "axioma-dev".to_string(),
        "--display-name".to_string(),
        "Axioma Dev".to_string(),
    ])
    .expect("install args");
    match install {
        CliCommand::Install(options) => {
            assert!(options.user);
            assert_eq!(options.kernel_name, "axioma-dev");
            assert_eq!(options.display_name, "Axioma Dev");
        }
        other => panic!("unexpected command: {other:?}"),
    }

    let run = parse_cli_command(&[
        "axioma-jupyter".to_string(),
        "connection.json".to_string(),
    ])
    .expect("run args");
    match run {
        CliCommand::Run { connection_file } => {
            assert_eq!(connection_file, PathBuf::from("connection.json"));
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[test]
fn startup_config_prefers_explicit_workdir_env() {
    let temp = temp_dir("startup-config");
    let connection = temp.join("connection.json");
    fs::write(&connection, "{}").expect("write connection");
    let config = resolve_startup_config_from_values(
        &connection,
        Some(temp.join("notebooks").into_os_string()),
        Some(PathBuf::from("/tmp/std").into_os_string()),
        temp.join("fallback"),
    );
    assert_eq!(config.connection_file, connection);
    assert_eq!(config.working_dir, temp.join("notebooks"));
    assert_eq!(config.env_std_path, Some(PathBuf::from("/tmp/std").into_os_string()));
}
