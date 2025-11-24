#define Version Trim(FileRead(FileOpen("..\VERSION")))
#define MatrixName GetEnv('MATRIX_NAME')
#define ProjectName GetEnv('PROJECT_NAME')
#define ProductName GetEnv('PRODUCT_NAME')
#define Publisher GetEnv('COMPANY_NAME')
#define Year GetDateTimeString("yyyy","","")

; 'Types': What get displayed during the setup
[Types]
Name: "full"; Description: "Full installation"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

; Components are used inside the script and can be composed of a set of 'Types'
[Components]
Name: "vst3"; Description: "VST3 plugin"; Types: full custom

[Setup]
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
PrivilegesRequired=lowest
AppName={#ProductName}
AppVerName={#ProductName} {#MatrixName} {#Version}
OutputBaseFilename={#ProductName}-{#Version}-{#MatrixName}
AppCopyright=Copyright (C) {#Year} {#Publisher}
AppPublisher={#Publisher}
AppVersion={#Version}
DefaultDirName="{userappdata}\REAPER\Plugins\{#ProductName}.vst3"
DisableDirPage=no

; MAKE SURE YOU READ/MODIFY THE EULA BEFORE USING IT
; LicenseFile="resources\EULA"
UninstallFilesDir="{localappdata}\{#ProductName}\uninstall"

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\REAPER\Plugins\{#ProductName}Data"

; MSVC adds a .ilk when building the plugin. Let's not include that.
; The parakeet-transcribe-windows.exe is already in the VST3 bundle at Contents/x86_64-win
; so it will be included by the recursesubdirs flag below
[Files]
Source: "..\Builds\{#ProjectName}_artefacts\Release\VST3\{#ProductName}.vst3\*"; DestDir: "{app}"; Excludes: *.ilk; Flags: ignoreversion recursesubdirs; Components: vst3

[Icons]
Name: "{autoprograms}\Uninstall {#ProductName}"; Filename: "{uninstallexe}"
