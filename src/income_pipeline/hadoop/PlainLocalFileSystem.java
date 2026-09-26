package income.pipeline.hadoop;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RawLocalFileSystem;

/**
 * Local filesystem whose directory listing uses {@link File#list()}.
 *
 * Hadoop's {@code FileUtil.list} asks {@code NativeIO.Windows.access} for
 * every entry. That native call needs {@code hadoop.dll}, which a normal
 * Windows JDK does not ship. Model save reaches that call while committing
 * task output, so the stock local filesystem cannot finish the write.
 */
public final class PlainLocalFileSystem extends LocalFileSystem {
  public PlainLocalFileSystem() {
    super(new PlainRawLocalFileSystem());
  }
}

final class PlainRawLocalFileSystem extends RawLocalFileSystem {
  @Override
  public FileStatus[] listStatus(Path path) throws IOException {
    File local = pathToFile(path);
    if (!local.exists()) {
      throw new FileNotFoundException(path.toString());
    }
    if (!local.isDirectory()) {
      return new FileStatus[] {getFileStatus(path)};
    }
    String[] names = local.list();
    if (names == null) {
      throw new IOException("cannot list " + local);
    }
    List<FileStatus> statuses = new ArrayList<FileStatus>(names.length);
    for (String name : names) {
      try {
        statuses.add(getFileStatus(new Path(path, name)));
      } catch (FileNotFoundException ignored) {
        // The entry disappeared between list and stat.
      }
    }
    return statuses.toArray(new FileStatus[0]);
  }
}
