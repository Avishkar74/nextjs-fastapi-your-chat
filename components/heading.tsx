import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface HeadingProps {
  title: string;
  description: string;
  icon?: LucideIcon;
  iconColor?: string;
  bgColor?: string;
  className?: string;
}

export const Heading = ({
  title,
  description,
  icon: Icon,
  iconColor,
  bgColor,
  className,
}: HeadingProps) => {
  return (
    <div className={cn("text-center", className)}>
      {Icon && (
        <div className={cn("inline-flex p-3 rounded-xl mb-4", bgColor)}>
          <Icon className={cn("w-8 h-8", iconColor)} />
        </div>
      )}
      <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-3">
        {title}
      </h1>
      <p className="text-lg text-gray-600 max-w-3xl mx-auto leading-relaxed">
        {description}
      </p>
    </div>
  );
};
