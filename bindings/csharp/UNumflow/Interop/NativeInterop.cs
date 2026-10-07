using System.Runtime.InteropServices;

namespace UNumflow.Interop;

/// <summary>The C ABI of u-numflow: every call takes a JSON request and writes a JSON body.</summary>
internal static partial class NativeInterop
{
    private const string DllName = "u_numflow";

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_distribution_moments(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_distribution_cdf(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_distribution_quantile(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_distribution_sample(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_interval_normalize(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_interval_measure(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_interval_contains(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_interval_union(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_interval_intersection(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int unumflow_interval_difference(string requestJson, out IntPtr resultPtr);

    [LibraryImport(DllName)]
    public static partial void unumflow_free_string(IntPtr ptr);

    [LibraryImport(DllName)]
    public static partial IntPtr unumflow_version();
}
